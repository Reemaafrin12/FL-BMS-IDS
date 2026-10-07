import os
from nfstream import NFStreamer
import pandas as pd

# Specify the input directory containing pcap files
input_directory = '/.../input/evse-pcaps/'

# Specify the output directory for CSV files
output_folder = '/.../csv_files/'
os.makedirs(output_folder, exist_ok=True)

# Specify the maximum number of packets to process from each PCAP file
max_packets = 1000 # apply as required

# Loop through each file in the input directory
for pcap_file in os.listdir(input_directory):
    if pcap_file.endswith('.pcap'):
        input_file_path = os.path.join(input_directory, pcap_file)
        output_file_path = os.path.join(output_folder, os.path.splitext(pcap_file)[0] + '.csv')

        # Initialize variables
        packets_processed = 0

        # Create NFStreamer object
        nfstreamer = NFStreamer(source=input_file_path, statistical_analysis=True)

        # Convert pcap to DataFrame using NFStreamer
        for flow in nfstreamer:
            if packets_processed >= max_packets:
                break  # Exit the loop if the maximum number of packets is reached

            # Process the flow if needed
            # ...

            packets_processed += 1

        # Save the DataFrame to CSV
        nfstreamer.to_pandas().to_csv(output_file_path, index=False)

        print(f"File converted: {output_file_path}")