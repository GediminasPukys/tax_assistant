import requests
import os
from urllib.parse import urlparse
from pathlib import Path


def save_html_from_urls(urls_file, output_dir='VMI_HTMLS'):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read URLs from file
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Process each URL
    for url in urls:
        try:
            # Send GET request
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = parsed_url.netloc + parsed_url.path
            filename = filename.replace('/', '_')
            if not filename.endswith('.html'):
                filename += '.html'

            # Save HTML content
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print(f"Successfully saved {url} to {filepath}")

        except requests.RequestException as e:
            print(f"Error downloading {url}: {str(e)}")
        except IOError as e:
            print(f"Error saving {url}: {str(e)}")


if __name__ == "__main__":
    save_html_from_urls('urls.txt')