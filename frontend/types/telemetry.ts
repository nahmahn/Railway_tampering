export interface TelemetryPackage {
    source_file_attributes: {
        filename: string;
        file_format: string;
        resolution: string;
        color_space: string;
        file_size_mb?: number;
    };
    acquisition_timestamp: {
        utc_time: string;
        unix_epoch_ns: number;
        sync_source: string;
    };
    sensor_hardware_config: {
        sensor_id: string;
        camera_model: string;
        lens_type: string;
        exposure_time_us: number;
        analog_gain_db: number;
        aperture_f_stop: number;
        focus_mode: string;
    };
    geo_positioning_input: {
        device: string;
        latitude_decimal: number;
        longitude_decimal: number;
        altitude_msl_m: number;
        satellite_count: number;
        hdop: number;
    };
    locomotive_telemetry_bus: {
        loco_id: string;
        train_interface_unit_status: string;
        current_speed_kmh: number;
        throttle_notch: number;
        brake_cylinder_pressure_kg_cm2: number;
        heading_magnetic: number | null;
    };
    environmental_sensor_readings: {
        ambient_light_lux: number;
        external_temp_c: number;
        humidity_percent: number;
    };
}
