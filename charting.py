import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot regularity scores vs frame numbers from CSV")
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with scores')
    parser.add_argument('--video', type=str, default=None, help='Filter by specific video file')
    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} does not exist")
        return

    # Read the CSV
    try:
        df = pd.read_csv(args.csv, sep=',')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Verify required columns
    required_columns = ['clip_start_frame', 'regularity_score']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter by video if specified
    if args.video:
        df = df[df['video'] == args.video]
        if df.empty:
            print(f"No data found for video: {args.video}")
            return

    # Extract data and convert to native Python types
    try:
        scores = df['regularity_score'].astype(float).tolist()
        frames = df['clip_start_frame'].astype(int).tolist()
    except ValueError as e:
        print(f"Error converting data to numeric values: {e}")
        return

    # Create scatter plot with matplotlib (swapped axes)
    plt.figure(figsize=(10, 6))
    plt.scatter(frames, scores, color='teal', alpha=0.7, s=50)
    plt.xlabel('Clip Start Frame')
    plt.ylabel('Regularity Score')
    plt.title(f'grafik video ({args.video if args.video else "lab"})')
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()