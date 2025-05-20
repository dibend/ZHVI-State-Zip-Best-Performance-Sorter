import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import pyopencl as cl
import os
import warnings

# Optional: Suppress PyOpenCL specific warnings if needed
# warnings.filterwarnings("ignore", message=".*kernel caching.*", module='pyopencl')

# --- Constants ---
ZILLOW_DATA_URL = 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
MIN_YEAR = 2000
DEFAULT_STATE = "NJ" # Default state
DEFAULT_START_DATE = "2014-01-31" # Use month-end dates often present in Zillow data
DEFAULT_END_DATE = "2023-12-31"
NAN_MARKER = -99999.0 # Marker for missing data passed to OpenCL

# --- OpenCL Kernel for Growth Calculation ---
# Calculates percentage growth = (end_price - start_price) / start_price
growth_kernel_code = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void calculate_growth(
    __global const double* start_prices, // Input: Start prices for each ZIP
    __global const double* end_prices,   // Input: End prices for each ZIP
    __global double* results,          // Output: Growth percentage for each ZIP
    const unsigned int num_zips,        // Input: Total number of zip codes
    const double nan_marker            // Input: Marker used for missing start/end price
) {
    unsigned int zip_idx = get_global_id(0); // Index of the current ZIP code

    if (zip_idx >= num_zips) {
        return; // Boundary check
    }

    double start_price = start_prices[zip_idx];
    double end_price = end_prices[zip_idx];

    // Check if start or end price is missing (using the marker) or non-positive
    if (start_price <= 0.0 || start_price == nan_marker || end_price <= 0.0 || end_price == nan_marker) {
        results[zip_idx] = nan_marker; // Mark result as invalid
    } else {
        // Calculate percentage growth
        results[zip_idx] = (end_price - start_price) / start_price;
    }
}
"""

# --- Data Loading Cache & State Extraction ---
zillow_df_cache = None
cache_load_time = None
us_states_cache = None

def load_zillow_data_and_states(max_age_hours=24):
    """Loads Zillow data from URL or cache, and extracts unique US states."""
    global zillow_df_cache, cache_load_time, us_states_cache
    now = time.time()
    use_cache = False
    if zillow_df_cache is not None and cache_load_time is not None:
        age_seconds = now - cache_load_time
        if age_seconds < max_age_hours * 3600:
            print("Using cached Zillow data and states.")
            use_cache = True

    if use_cache:
        return zillow_df_cache.copy(), us_states_cache
    else:
        print(f"Loading Zillow data from {ZILLOW_DATA_URL}...")
        try:
            df = pd.read_csv(ZILLOW_DATA_URL)
            # IMPORTANT: Ensure RegionName is string and padded
            df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)

            if 'State' in df.columns:
                 states = sorted(list(df['State'].dropna().unique()))
            else:
                 print("Warning: 'State' column not found.")
                 states = []

            zillow_df_cache = df
            us_states_cache = states
            cache_load_time = now
            print(f"Zillow data loaded. Found {len(states)} states/territories.")
            return df.copy(), states
        except Exception as e:
            print(f"Error loading Zillow data: {e}")
            raise gr.Error(f"Failed to load data from Zillow URL. Error: {e}")

# --- OpenCL Setup ---
def get_opencl_context_queue():
    """Initializes and returns a PyOpenCL context and command queue (cached)."""
    if hasattr(get_opencl_context_queue, "cache"):
        return get_opencl_context_queue.cache
    print("Initializing OpenCL context...")
    try:
        platform = cl.get_platforms()[0]
        devices = []
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            print(f"Using GPU: {devices[0].name}")
        except (cl.RuntimeError, IndexError):
            print("No GPU found or usable, falling back to CPU.")
            try:
                devices = platform.get_devices(device_type=cl.device_type.CPU)
                print(f"Using CPU: {devices[0].name}")
            except (cl.RuntimeError, IndexError):
                raise RuntimeError("No OpenCL devices found (GPU or CPU).")

        context = cl.Context(devices=[devices[0]])
        queue = cl.CommandQueue(context)
        get_opencl_context_queue.cache = (context, queue)
        return context, queue
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenCL: {e}")

# --- OpenCL Growth Calculation ---
def calculate_growth_opencl(context, queue, df_zillow, zip_codes, start_date_str, end_date_str):
    """
    Calculates percentage growth for multiple ZIP codes using OpenCL.

    Args:
        context: PyOpenCL context.
        queue: PyOpenCL command queue.
        df_zillow: Full Zillow DataFrame.
        zip_codes: List of ZIP codes (as strings) to process.
        start_date_str: Start date string (YYYY-MM-DD).
        end_date_str: End date string (YYYY-MM-DD).

    Returns:
        tuple: (pd.DataFrame, str, str) containing:
               - DataFrame with ZIP codes index and 'GrowthPercentage'.
               - The actual start date column name used.
               - The actual end date column name used.
               Returns empty DataFrame if insufficient data.
    """
    print(f"Preparing data for OpenCL growth calculation: {len(zip_codes)} ZIPs...")

    # --- 1. Find Exact Start/End Date Columns Available ---
    try:
        target_start_dt = pd.to_datetime(start_date_str)
        target_end_dt = pd.to_datetime(end_date_str)
    except ValueError:
        raise ValueError("Invalid start or end date format.")

    first_date_col_index = -1
    all_date_cols_dt = {}
    for i, col_name in enumerate(df_zillow.columns):
        if isinstance(col_name, str) and (col_name.count('-') == 1 or col_name.count('-') == 2): # Zillow uses YYYY-MM or YYYY-MM-DD
            try:
                # Attempt to parse, be flexible with day part if missing (assume first of month)
                if col_name.count('-') == 1: # YYYY-MM
                    dt_candidate = pd.to_datetime(col_name + '-01', errors='raise')
                else: # YYYY-MM-DD
                    dt_candidate = pd.to_datetime(col_name, errors='raise')

                all_date_cols_dt[col_name] = dt_candidate
                if first_date_col_index == -1: first_date_col_index = i
            except (ValueError, TypeError):
                # print(f"Debug: Could not parse column '{col_name}' as date.") # Optional debug
                continue
    if first_date_col_index == -1: raise ValueError("Could not identify any date columns in the Zillow data.")


    actual_start_col, actual_end_col = None, None
    min_start_diff, min_end_diff = pd.Timedelta.max, pd.Timedelta.max

    for col, dt in all_date_cols_dt.items():
        if dt >= target_start_dt:
            diff = dt - target_start_dt
            if diff < min_start_diff:
                min_start_diff, actual_start_col = diff, col
            # If multiple columns have the same minimum difference (e.g. same day), prefer the one that is also closest to the end of its month if data is EOM
            elif diff == min_start_diff and dt.is_month_end:
                 actual_start_col = col


        if dt <= target_end_dt:
            diff = target_end_dt - dt
            if diff < min_end_diff:
                min_end_diff, actual_end_col = diff, col
            elif diff == min_end_diff and dt.is_month_end:
                actual_end_col = col


    if actual_start_col is None or actual_end_col is None:
        err_msg = "Could not find suitable date columns. "
        if actual_start_col is None: err_msg += f"No data found on or after start date {start_date_str}. "
        if actual_end_col is None: err_msg += f"No data found on or before end date {end_date_str}. "
        raise ValueError(err_msg)

    if all_date_cols_dt[actual_start_col] > all_date_cols_dt[actual_end_col]:
        raise ValueError(f"Selected start date '{actual_start_col}' ({all_date_cols_dt[actual_start_col]:%Y-%m-%d}) is after selected end date '{actual_end_col}' ({all_date_cols_dt[actual_end_col]:%Y-%m-%d}). Try a later start or earlier end date.")

    print(f"Using actual data columns: Start='{actual_start_col}', End='{actual_end_col}'")

    # --- 2. Prepare Price Data for OpenCL ---
    # Ensure we filter using the string ZIP codes
    df_state = df_zillow[df_zillow['RegionName'].isin(zip_codes)].copy()
    # Select only the ZIP code and the two relevant date columns
    price_data = df_state[['RegionName', actual_start_col, actual_end_col]].set_index('RegionName')
    # Reindex to ensure all requested zips are present and in order (index is already strings)
    price_data = price_data.reindex(zip_codes)

    num_zips = len(price_data)
    if num_zips == 0: return pd.DataFrame(), actual_start_col, actual_end_col

    # Extract start and end prices, fill missing with NAN_MARKER
    start_prices_np = price_data[actual_start_col].fillna(NAN_MARKER).values.astype(np.float64)
    end_prices_np = price_data[actual_end_col].fillna(NAN_MARKER).values.astype(np.float64)
    # Output array for growth results
    results_np = np.empty(num_zips, dtype=np.float64)

    print(f"Data prepared for {num_zips} ZIPs.")

    # --- 3. Setup OpenCL Buffers ---
    mf = cl.mem_flags
    try:
        start_prices_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=start_prices_np)
        end_prices_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=end_prices_np)
        results_buf = cl.Buffer(context, mf.WRITE_ONLY, results_np.nbytes)
    except cl.Error as e:
        raise RuntimeError(f"Failed to create OpenCL buffers. CL Error Code: {e.code}")

    # --- 4. Compile & Execute Kernel ---
    try:
        # Use function attribute as a simple program cache
        cache_key = "growth_kernel"
        if not hasattr(calculate_growth_opencl, "program_cache"):
            calculate_growth_opencl.program_cache = {}
        if cache_key not in calculate_growth_opencl.program_cache:
            print("Compiling growth calculation kernel...")
            program = cl.Program(context, growth_kernel_code).build()
            calculate_growth_opencl.program_cache[cache_key] = program
            print("Kernel compiled.")
        else:
            program = calculate_growth_opencl.program_cache[cache_key]

        kernel = program.calculate_growth
        # Set scalar arg types
        kernel.set_scalar_arg_dtypes([None, None, None, np.uint32, np.float64])

        global_size = (num_zips,) # One work item per ZIP code
        local_size = None # Let OpenCL decide

        print(f"Executing growth kernel for {num_zips} ZIPs...")
        start_time = time.time()
        kernel_event = kernel(queue, global_size, local_size,
                              start_prices_buf, end_prices_buf, results_buf,
                              np.uint32(num_zips), np.float64(NAN_MARKER))
        kernel_event.wait()
        end_time = time.time()
        print(f"OpenCL growth calculation finished in {end_time - start_time:.3f} seconds.")

    except cl.Error as e:
        build_log = "Build log not available."
        try: build_log = program.get_build_info(context.devices[0], cl.program_build_info.LOG)
        except: pass
        raise RuntimeError(f"OpenCL Kernel Error (Build/Exec). Code: {e.code}\nLog:\n{build_log}")

    # --- 5. Retrieve Results ---
    try:
        cl.enqueue_copy(queue, results_np, results_buf).wait()
    except cl.Error as e:
        raise RuntimeError(f"Failed to copy results from device. CL Error Code: {e.code}")

    # --- 6. Format Results ---
    # Create DataFrame with results, using the original string index from price_data
    results_df = pd.DataFrame({'GrowthPercentage': results_np}, index=price_data.index)
    # Replace the marker with actual NaN for pandas operations
    results_df['GrowthPercentage'] = results_df['GrowthPercentage'].replace(NAN_MARKER, np.nan)
    # Drop rows where growth couldn't be calculated
    results_df = results_df.dropna()

    print(f"Formatted growth results for {len(results_df)} ZIPs.")
    return results_df, actual_start_col, actual_end_col


# --- Analysis and Plotting ---
def analyze_and_plot(selected_state, start_date_str, end_date_str):
    """
    Main function: Loads data, runs OpenCL growth calculation, generates plots.
    """
    print("-" * 30)
    print(f"Request: State='{selected_state}', Start='{start_date_str}', End='{end_date_str}'")

    # --- Input Validation ---
    try:
        # Basic validation, more happens in calculate_growth_opencl
        pd.to_datetime(start_date_str); pd.to_datetime(end_date_str)
        if not selected_state: raise ValueError("State not selected.")
    except Exception as e:
        raise gr.Error(f"Invalid input format or range. Error: {e}")

    # --- Load Data & Get ZIPs ---
    df_zillow, _ = load_zillow_data_and_states()
    state_zips_df = df_zillow[df_zillow['State'] == selected_state]
    if state_zips_df.empty:
        raise gr.Error(f"No data found for state '{selected_state}'.")
    # Get unique list of string ZIP codes
    zip_codes_to_analyze = state_zips_df['RegionName'].unique().tolist()
    print(f"Found {len(zip_codes_to_analyze)} unique ZIP codes for state {selected_state}.")
    if not zip_codes_to_analyze:
         raise gr.Error(f"No ZIP codes found for {selected_state}.")

    # --- Calculate Growth using OpenCL ---
    try:
        context, queue = get_opencl_context_queue()
        results_df, actual_start, actual_end = calculate_growth_opencl(
            context, queue, df_zillow, zip_codes_to_analyze, start_date_str, end_date_str
        )
    except Exception as e: # Catch specific ValueErrors from date finding too
        print(f"Error during OpenCL calculation or date finding: {e}") # Log for debugging
        raise gr.Error(f"Growth calculation failed: {e}")


    if results_df.empty:
        raise gr.Error(f"No valid growth data calculated for any ZIP in {selected_state} between {start_date_str} and {end_date_str} (using actual dates {actual_start} to {actual_end}). This could be due to lack of data for the period or issues with selected dates.")

    # --- Create Histogram Plot ---
    print("Generating growth histogram...")
    fig_hist = go.Figure(data=[go.Histogram(
        x=results_df['GrowthPercentage'] * 100, # Convert to percentage for plot
        name='Growth Distribution',
        nbinsx=50 # Adjust number of bins as needed
    )])

    # Add summary statistics lines
    mean_growth = results_df['GrowthPercentage'].mean() * 100
    median_growth = results_df['GrowthPercentage'].median() * 100
    p10 = results_df['GrowthPercentage'].quantile(0.10) * 100
    p90 = results_df['GrowthPercentage'].quantile(0.90) * 100

    # MODIFIED SECTION FOR ANNOTATION POSITIONING
    fig_hist.add_vline(x=p10, line_dash="dash", line_color="yellow",
                        annotation_text=f"10th Perc: {p10:.1f}%",
                        annotation_position="top left")

    fig_hist.add_vline(x=median_growth, line_dash="dash", line_color="red",
                        annotation_text=f"Median: {median_growth:.1f}%",
                        annotation_position="bottom right") # Changed from "top" to avoid mean if close

    fig_hist.add_vline(x=p90, line_dash="dash", line_color="yellow",
                        annotation_text=f"90th Perc: {p90:.1f}%",
                        annotation_position="top right")

    fig_hist.add_vline(x=mean_growth, line_dash="dot", line_color="cyan",
                        annotation_text=f"Mean: {mean_growth:.1f}%",
                        annotation_position="top") # Kept as "top", usually distinct enough from P10/P90 edges

    fig_hist.update_layout(
        title=f'{selected_state} ZHVI Growth Distribution<br><sup>({actual_start} to {actual_end})</sup>',
        xaxis_title='Total Growth Percentage (%)',
        yaxis_title='Number of ZIP Codes',
        template="plotly_dark"
    )
    print("Histogram generated.")

    # --- Create Summary Table ---
    # Sort by growth for display
    summary_df = results_df.sort_values(by='GrowthPercentage', ascending=False)
    # Format as percentage string
    summary_df['GrowthPercentage'] = summary_df['GrowthPercentage'].map('{:.1%}'.format)
    print("Summary table prepared.")

    # Limit rows for display
    max_rows_display = 1000
    if len(summary_df) > max_rows_display:
        print(f"Displaying first {max_rows_display} rows of summary table.")
        summary_df_display = summary_df.head(max_rows_display)
    else:
        summary_df_display = summary_df

    # Reset index so ZIP code becomes a column for Gradio display
    summary_df_display = summary_df_display.reset_index().rename(columns={'RegionName': 'ZIP Code'})

    return fig_hist, summary_df_display


# --- Load initial data for UI setup ---
try:
    _, initial_us_states = load_zillow_data_and_states()
except Exception as e:
    print(f"CRITICAL ERROR: Could not load initial Zillow data for UI setup. {e}")
    initial_us_states = ["Error loading states"] # Placeholder

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="cyan", secondary_hue="teal"), title="State-Level Zillow ZIP Growth") as demo:
    gr.Markdown("# State-Level Zillow ZHVI Growth Analysis (OpenCL Accelerated)")
    gr.Markdown(
        "Select a US State and date range to calculate the total percentage growth of the Zillow Home Value Index (ZHVI) "
        "for **all** available ZIP codes in that state. The growth calculation is accelerated using **PyOpenCL**."
        "\n*Data Source: Zillow Research - [ZHVI Data](https://www.zillow.com/research/data/)*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            # State Selection
            state_input = gr.Dropdown(
                choices=initial_us_states,
                label="Select US State",
                value=DEFAULT_STATE
            )
            # Date Inputs
            start_date_input = gr.Textbox(label="Start Date", value=DEFAULT_START_DATE, placeholder="YYYY-MM-DD or YYYY-MM")
            end_date_input = gr.Textbox(label="End Date", value=DEFAULT_END_DATE, placeholder="YYYY-MM-DD or YYYY-MM")
            run_button = gr.Button("Calculate State Growth", variant="primary")

        with gr.Column(scale=2):
             # Output for the summary table
             summary_output = gr.DataFrame(
                 label="ZIP Code Growth Summary (Top 1000)",
                 interactive=False,
                 # For Gradio 4.x and later, headers are typically inferred or can be controlled differently
                 # For older versions, ensure it's a list of strings:
                 # headers=['ZIP Code', 'GrowthPercentage'] # This might be needed for specific Gradio versions
             )
             gr.Markdown("*Table shows calculated growth for each valid ZIP, sorted descending. May show a sample if >1000 ZIPs analyzed.*")


    with gr.Row():
        # Output for the plot
        plot_output = gr.Plot(label="Distribution of Growth Rates")

    # Connect button click to function
    run_button.click(
        fn=analyze_and_plot,
        # Inputs must match the function signature order
        inputs=[state_input, start_date_input, end_date_input],
        outputs=[plot_output, summary_output] # Plot first, then table
    )

    # Provide examples relevant to state analysis
    gr.Examples(
        examples=[
            ["NJ", "2015-01-31", "2023-12-31"],
            ["CA", "2018-01", "2024-03"], # Example with YYYY-MM
            ["FL", "2012-01-31", "2022-12-31"],
            ["TX", "2016-06", "2023-06-30"], # Mixed example
            ["CO", "2019-01-31", "2024-03-31"],
        ],
        # Ensure example inputs match the function signature order
        inputs=[state_input, start_date_input, end_date_input]
    )

# --- Launch App ---
if __name__ == "__main__":
    # Check for siphash warning fix
    try:
        import siphash24
    except ImportError:
        print("\nWarning: For potential minor performance improvement in PyOpenCL caching, run:")
        print("pip install siphash24\n")

    demo.launch(share=True, debug=True)
