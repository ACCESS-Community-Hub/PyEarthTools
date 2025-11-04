import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # A spot to put the data on disk. We keep both the data as-downloaded and the reprocessed version, so you might need up to 50GB free in order to make this work.

    import requests
    from pathlib import Path
    from tqdm.auto import tqdm

    DOWNLOAD_DIR = Path.home() / 'hadisd' / 'as_downloaded'  # We will download data here and keep a copy

    # For testing, we download just under 4GB data
    testing_download = [
        "000000-029999", "500000-549999", "722000-722999", "800000-849999",
    ]

    # Download list for all files
    full_download = [
        "000000-029999", "030000-049999", "050000-079999", "080000-099999",
        "100000-149999", "150000-199999", "200000-249999", "250000-299999",
        "300000-349999", "350000-399999", "400000-449999", "450000-499999",
        "500000-549999", "550000-599999", "600000-649999", "650000-699999", 
        "700000-709999", "710000-714999", "715000-719999", "720000-721999",
        "722000-722999", "723000-723999", "724000-724999", "725000-725999", 
        "726000-726999", "727000-729999", "730000-799999", "800000-849999",
        "850000-899999", "900000-949999", "950000-999999",
    ]
    return DOWNLOAD_DIR, full_download, requests, tqdm


@app.cell
def _(requests, tqdm):
    def download_wmo_range(wmo_id_range, download_dir):
        wmo_str = f"WMO_{wmo_id_range}"
        url = f"https://www.metoffice.gov.uk/hadobs/hadisd/v343_2025f/data/{wmo_str}.tar.gz"
        tar_name = f"{wmo_str}.tar.gz"
        filename = download_dir / tar_name    

        head = requests.head(url, allow_redirects=True)
        remote_size = int(head.headers.get('content-length', 0))
        local_size = filename.stat().st_size if filename.exists() else 0

        if filename.exists() and local_size == remote_size:
            print(f"File already fully downloaded: {filename} ({local_size/1024**2:.2f} MB)")
        elif filename.exists() and local_size != remote_size:
            # Users may have done this deliberately, so just print a message
            print(f"Local filesize of {filename} does not match, please delete it and re-download it")
        else:
            headers = {}
            mode = 'wb'
            initial_pos = 0
            if filename.exists() and local_size < remote_size:
                headers['Range'] = f'bytes={local_size}-'
                mode = 'ab'
                initial_pos = local_size
                print(f"Resuming download for {filename.name} at {local_size/1024**2:.2f} MB...")
            else:
                print(f"Starting download for {filename.name}...")

            response = requests.get(url, stream=True, headers=headers)
            total = remote_size
            with open(filename, mode) as f, tqdm(
                desc=f"Downloading {filename.name}",
                total=total,
                initial=initial_pos,
                unit='B', unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            final_size = filename.stat().st_size
            if final_size == remote_size:
                print(f"Download complete: {filename} ({final_size/1024**2:.2f} MB)")
            else:
                print(f"Warning: Download incomplete. Local size: {final_size}, Remote size: {remote_size}")

        return filename, tar_name
    return (download_wmo_range,)


@app.cell
def _(DOWNLOAD_DIR, download_wmo_range, full_download):
    # for wrange in testing_download:
    #     download_wmo_range(wrange, DOWNLOAD_DIR)

    # FOR FULL STATION DOWNLOAD
    # Note, if at NCI doing the hackathon, use the pre-downloaded data

    for wrange in full_download:
        download_wmo_range(wrange, DOWNLOAD_DIR)    
    return


@app.cell
def _():
    # The next step is easiest to do manually, and is a bit awkward to put in a notebook step.

    # First, go to your top-level download directory. Make a new directory called 'unpacked', then run the following command.
    # This will result in a lot of individual .nc.gz files on disK

    # `for file in *.tar.gz; do tar -xzf "$file" --directory ../unpacked; done`

    # Once this is down, change directory into the unpacked directory and run

    # `gunzip *`

    # This is much faster for some reason than trying to use Python to get the job done.
    return


@app.cell
def _():
    print("download completed)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
