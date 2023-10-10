import hydro_processing_tools.data_acquisition as data_acquisition
import hydro_processing_tools.smooth as smooth
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_url = "http://tsdata.horizons.govt.nz/"
    hts = "Toha.hts"
    site = "Manawatu at Teachers College"
    measurement = "Flow"
    from_date = "2012-01-22 10:50"
    to_date = "2018-04-13 14:05"
    dtl_method = "trend"

    # Acquire the data
    data = data_acquisition.get_data(base_url, hts, site, measurement, from_date, to_date, dtl_method)
    
    # Perform spike removal using your 'remove_spikes' function
    span = 10
    high_clip = 1000
    low_clip = -1000
    delta = 500
    cleaned_data = smooth.remove_spikes(data["Value"], span, high_clip, low_clip, delta)
    print(cleaned_data)

    # Plot the data before and after spike removal
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data, label="Original Data")
    plt.title("Data Before Spike Removal")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(cleaned_data, label="Cleaned Data", color='orange')
    plt.title("Data After Spike Removal")
    plt.legend()

    plt.tight_layout()
    plt.show()
