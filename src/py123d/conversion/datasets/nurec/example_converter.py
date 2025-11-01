"""Test script for NUREC converter."""

from pathlib import Path

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.nurec.nurec_converter import NURECConverter
from py123d.conversion.log_writer.arrow_log_writer import ArrowLogWriter
from py123d.conversion.map_writer.gpkg_map_writer import GPKGMapWriter

# Paths
SOURCE_PATH = Path("/home/bastian/data/nvidia/sample_set/test_conversion")
TARGET_PATH = Path("/home/bastian/dev/123d/target")

# Configuration
config = DatasetConverterConfig(
    include_map=True,
    include_ego=True,  # Enable ego state writing
    include_box_detections=True,  # Enable detection writing
    include_scenario_tags=True,  # Enable scenario tags
    include_cameras=True,  # Enable cameras
    camera_store_option="path",  # Store as path+frame_index for now
    include_lidars=False,  # nurec doesn't have LiDAR data
    force_map_conversion=False,  # Don't reconvert maps
    force_log_conversion=True,  # Reconvert logs to test cameras
)

# Create converter
converter = NURECConverter(
    splits=["nurec_batch0002"],  # Only tested on single batch as of now
    nurec_data_root=SOURCE_PATH,
    dataset_converter_config=config,
)

print(f"Number of maps: {converter.get_number_of_maps()}")
print(f"Number of logs: {converter.get_number_of_logs()}")

# Convert maps
print("\n=== Converting Maps ===")
map_writer = GPKGMapWriter(TARGET_PATH / "maps")
for map_idx in range(converter.get_number_of_maps()):
    print(f"Converting map {map_idx + 1}/{converter.get_number_of_maps()}...")
    converter.convert_map(map_idx, map_writer)

# Convert logs
print("\n=== Converting Logs ===")
log_writer = ArrowLogWriter(TARGET_PATH / "logs")
for log_idx in range(converter.get_number_of_logs()):
    print(f"Converting log {log_idx + 1}/{converter.get_number_of_logs()}...")
    converter.convert_log(log_idx, log_writer)

print("\n=== Conversion Complete ===")
