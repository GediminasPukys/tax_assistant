import requests
import os
import csv
from pathlib import Path
import pandas as pd


def save_html_from_csv(csv_file, output_dir='vmi_docs'):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Read CSV file using pandas
        df = pd.read_csv(csv_file)

        # Process each URL and version
        for _, row in df.iterrows():
            url = row['url']
            version = row['version']

            try:
                # Send GET request
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Generate filename using version
                filename = f"{version}.html"
                filepath = os.path.join(output_dir, filename)

                # Save HTML content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                print(f"Successfully saved {version} to {filepath}")

            except requests.RequestException as e:
                print(f"Error downloading {version} ({url}): {str(e)}")
            except IOError as e:
                print(f"Error saving {version}: {str(e)}")

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")


def main():
    csv_file = 'vmi_docs.csv'
    save_html_from_csv(csv_file)


if __name__ == "__main__":
    main()