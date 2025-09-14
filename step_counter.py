
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.signal import detrend, butter, filtfilt
from scipy.ndimage import median_filter


class FreshStepCounter:
    """Fresh step counter from scratch."""

    def __init__(self):
        # Settings
        self.fps = 25.0
        # Data storage
        self.raw_data = None
        self.left_xyz = None
        self.right_xyz = None
        self.left_pc1_raw = None
        self.right_pc1_raw = None
        self.left_pc1_clean = None
        self.right_pc1_clean = None
        self.difference_signal = None
        self.step_frames = None
        self.total_steps = 0

    def load_csv(self, file_path):
        """Load CSV file."""
        print(f"Loading CSV: {file_path}")
        self.raw_data = pd.read_csv(file_path)
        self.raw_data.columns = self.raw_data.columns.str.strip()
        print(f"Loaded {len(self.raw_data)} frames")
        print("=" * 50)
        return self

    def extract_coordinates(self):
        """Extract X,Y,Z coordinates for both ankles."""
        print("EXTRACTING ANKLE COORDINATES")

        # Get raw coordinates
        left_x_raw = self.raw_data['Left AnkleX'].values
        left_y_raw = self.raw_data['Left AnkleY'].values
        left_z_raw = self.raw_data['Left AnkleDepth'].values

        right_x_raw = self.raw_data['Right AnkleX'].values
        right_y_raw = self.raw_data['Right AnkleY'].values
        right_z_raw = self.raw_data['Right AnkleDepth'].values

        # Clean coordinates (handle zeros and NaN)
        def clean_coordinate(coord_array):
            coords = coord_array.astype(float)
            coords[coords == 0] = np.nan  # Replace zeros with NaN

            # Fill NaN values
            coords_series = pd.Series(coords)
            coords_series = coords_series.fillna(method='ffill')  # Forward fill
            coords_series = coords_series.fillna(method='bfill')  # Backward fill
            coords_series = coords_series.fillna(coords_series.mean())  # Fill with mean if still NaN

            return coords_series.values

        # Clean all coordinates
        left_x = clean_coordinate(left_x_raw)
        left_y = clean_coordinate(left_y_raw)
        left_z = clean_coordinate(left_z_raw)

        right_x = clean_coordinate(right_x_raw)
        right_y = clean_coordinate(right_y_raw)
        right_z = clean_coordinate(right_z_raw)

        # Store as 3D arrays
        self.left_xyz = np.column_stack([left_x, left_y, left_z])
        self.right_xyz = np.column_stack([right_x, right_y, right_z])

        return self

    def compute_pca(self):
        """Apply PCA to get PC1 for each ankle."""
        print("COMPUTING PCA")

        # Left ankle PCA
        pca_left = PCA(n_components=1)  # We only need PC1
        left_pc1_result = pca_left.fit_transform(self.left_xyz)
        self.left_pc1_raw = left_pc1_result.flatten()
        left_variance = pca_left.explained_variance_ratio_[0]

        # Right ankle PCA
        pca_right = PCA(n_components=1)
        right_pc1_result = pca_right.fit_transform(self.right_xyz)
        self.right_pc1_raw = right_pc1_result.flatten()
        right_variance = pca_right.explained_variance_ratio_[0]

        return self

    def remove_nonstationarity(self):
        """Remove non-stationarity from PC1 signals."""
        print("REMOVING NON-STATIONARITY")

        def clean_signal(pc1_signal):
            """Clean a PC1 signal by removing trends."""
            # Step 1: Remove linear trend
            linear_clean = detrend(pc1_signal, type='linear')

            # Step 2: Remove quadratic trend
            time_index = np.arange(len(pc1_signal))
            quadratic_coeffs = np.polyfit(time_index, linear_clean, deg=2)
            quadratic_trend = np.polyval(quadratic_coeffs, time_index)
            quadratic_clean = linear_clean - quadratic_trend

            # Step 3: High-pass filter to remove very slow drift
            nyquist_freq = self.fps / 2.0
            cutoff_freq = 0.1  # Remove trends slower than 0.1 Hz (10+ seconds)

            if cutoff_freq < nyquist_freq:
                b, a = butter(2, cutoff_freq / nyquist_freq, btype='high')
                final_clean = filtfilt(b, a, quadratic_clean)
            else:
                final_clean = quadratic_clean

            return final_clean

        # Clean both PC1 signals
        self.left_pc1_clean = clean_signal(self.left_pc1_raw)
        self.right_pc1_clean = clean_signal(self.right_pc1_raw)

        return self

    def find_intersections(self):
        """Find intersections between left and right PC1."""
        print("FINDING PC1 INTERSECTIONS")

        # Calculate difference signal (Left - Right)
        self.difference_signal = self.left_pc1_clean - self.right_pc1_clean
        self.smooth_difference_signal = median_filter(np.asarray(self.difference_signal, float), size=9, mode="reflect")

        # Find zero crossings in difference signal
        zero_crossings = []
        for i in range(1, len(self.smooth_difference_signal)):
            # Check if signal crosses zero
            if ((self.smooth_difference_signal[i - 1] <= 0 and self.smooth_difference_signal[i] > 0) or
                    (self.smooth_difference_signal[i - 1] >= 0 and self.smooth_difference_signal[i] < 0)):
                zero_crossings.append(i)


        self.step_frames = zero_crossings
        self.total_steps = len(zero_crossings)
        return self

    def create_plots(self, output_file):
        """Create visualization plots."""
        print(f"CREATING PLOTS: {output_file}")

        # Create time axis
        frames = np.arange(len(self.raw_data))
        time_seconds = frames / self.fps

        # Create 4-panel subplot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Raw PC1 Signals (Left vs Right)',
                'PC1 After Non-Stationarity Removal (Left vs Right)',
                'Difference Signal (Left - Right) with Zero Crossings',
                'Smooth Difference Signal (Left - Right) with Zero Crossings',
                'Final Step Detection Results'
            ],
            vertical_spacing=0.08
        )

        # Panel 1: Raw PC1 signals
        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.left_pc1_raw,
            mode='lines', name='Left PC1 Raw',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.right_pc1_raw,
            mode='lines', name='Right PC1 Raw',
            line=dict(color='red', width=2)
        ), row=1, col=1)

        # Panel 2: Clean PC1 signals
        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.left_pc1_clean,
            mode='lines', name='Left PC1 Clean',
            line=dict(color='darkblue', width=2),
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.right_pc1_clean,
            mode='lines', name='Right PC1 Clean',
            line=dict(color='darkred', width=2),
            showlegend=False
        ), row=2, col=1)

        # Panel 3: Difference signal with zero crossings
        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.difference_signal,
            mode='lines', name='Difference (L-R)',
            line=dict(color='purple', width=2),
            showlegend=False
        ), row=3, col=1)

        # Panel 4: Smooth Difference signal with zero crossings
        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.smooth_difference_signal,
            mode='lines', name='Smooth Difference (L-R)',
            line=dict(color='red', width=2),
            showlegend=False
        ), row=3, col=1)

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # Mark zero crossings
        for step_frame in self.step_frames:
            step_time = step_frame / self.fps
            fig.add_vline(x=step_time, line_color="green", line_width=2, opacity=0.8, row=3, col=1)

        # Panel 5: Final results with step markers
        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.left_pc1_clean,
            mode='lines', name='Left Clean',
            line=dict(color='lightblue', width=1), opacity=0.7,
            showlegend=False
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=time_seconds, y=self.right_pc1_clean,
            mode='lines', name='Right Clean',
            line=dict(color='lightcoral', width=1), opacity=0.7,
            showlegend=False
        ), row=4, col=1)

        # Add step markers
        step_times = [f / self.fps for f in self.step_frames]
        for step_time in step_times:
            fig.add_vline(x=step_time, line_color="green", line_width=3, opacity=0.9, row=4, col=1)

        if step_times:
            fig.add_trace(go.Scatter(
                x=step_times, y=[0] * len(step_times),
                mode='markers', name=f'Steps ({self.total_steps})',
                marker=dict(color='green', size=12, symbol='triangle-down')
            ), row=4, col=1)

        # Update layout
        fig.update_layout(
            height=1200,
            title=f"Fresh Step Counter Results: {self.total_steps} Steps Detected",
            template="plotly_white"
        )

        # Update axes
        for i in range(1, 5):
            fig.update_xaxes(title_text="Time (seconds)" if i == 4 else "", row=i, col=1)

        fig.update_yaxes(title_text="PC1 Value", row=1, col=1)
        fig.update_yaxes(title_text="PC1 Value", row=2, col=1)
        fig.update_yaxes(title_text="Difference", row=3, col=1)
        fig.update_yaxes(title_text="Smooth Difference", row=3, col=1)
        fig.update_yaxes(title_text="PC1 Value", row=4, col=1)

        # Save plot
        fig.write_html(output_file)
        print(f"Plot saved: {output_file}")
        return self


    def print_summary(self):
        """Print final summary."""
        duration = len(self.raw_data) / self.fps

        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Video duration: {duration:.1f} seconds")
        print(f"Total steps detected: {self.total_steps}")

        print("=" * 50)
        return self


def analyze_steps(csv_file, output_html='fresh_results.html'):
    """Main function to run complete step analysis."""

    # Create step counter and run analysis
    counter = FreshStepCounter()

    # Run complete pipeline
    (counter
     .load_csv(csv_file)
     .extract_coordinates()
     .compute_pca()
     .remove_nonstationarity()
     .find_intersections()
     .create_plots(output_html)
     .print_summary())

    return counter.total_steps


# Run the analysis
if __name__ == "__main__":
    steps_detected = analyze_steps(
        '/Users/dmitrijsmirnov/Desktop/Projects/Python/DeepTrack3D/output_data/4_csv_data/test1_converted/test1_converted_segment_0s_to_16s_stabilized_analysis.csv')
