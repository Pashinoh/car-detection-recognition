CREATE TABLE `detection_log` (
  `id` int(11) NOT NULL,
  `timestamp` datetime NOT NULL,
  `class_name` varchar(50) NOT NULL,
  `track_id` int(11) NOT NULL,
  `source` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

CREATE TABLE `traffic_stats` (
  `id` int(11) NOT NULL,
  `total_car` int(11) NOT NULL DEFAULT 0,
  `total_motorcycle` int(11) NOT NULL DEFAULT 0,
  `total_bus` int(11) NOT NULL DEFAULT 0,
  `total_truck` int(11) NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;