-- phpMyAdmin SQL Dump
-- version 4.8.5
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 16, 2019 at 09:40 PM
-- Server version: 10.1.40-MariaDB
-- PHP Version: 7.3.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `grade`
--

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `Roll_no` int(3) NOT NULL,
  `GENRE` varchar(6) NOT NULL,
  `MARKS` int(3) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`Roll_no`, `GENRE`, `MARKS`) VALUES
(1, 'Male', 39),
(2, 'Male', 81),
(4, 'Female', 77),
(5, 'Female', 40),
(6, 'Female', 76),
(8, 'Female', 94),
(10, 'Female', 72),
(12, 'Female', 99),
(14, 'Female', 77),
(16, 'Male', 79),
(17, 'Female', 35),
(18, 'Male', 66),
(20, 'Female', 98),
(21, 'Male', 35),
(22, 'Male', 73),
(24, 'Male', 73),
(26, 'Male', 82),
(28, 'Male', 61),
(30, 'Female', 87),
(32, 'Female', 73),
(34, 'Male', 92),
(36, 'Female', 81),
(38, 'Female', 73),
(40, 'Female', 75),
(41, 'Female', 35),
(42, 'Male', 92),
(43, 'Male', 36),
(44, 'Female', 61),
(46, 'Female', 65),
(47, 'Female', 55),
(48, 'Female', 47),
(49, 'Female', 42),
(50, 'Female', 42),
(51, 'Female', 52),
(52, 'Male', 60),
(53, 'Female', 54),
(54, 'Male', 60),
(55, 'Female', 45),
(56, 'Male', 41),
(57, 'Female', 50),
(58, 'Male', 46),
(59, 'Female', 51),
(60, 'Male', 46),
(61, 'Male', 56),
(62, 'Male', 55),
(63, 'Female', 52),
(64, 'Female', 59),
(65, 'Male', 51),
(66, 'Male', 59),
(67, 'Female', 50),
(68, 'Female', 48),
(69, 'Male', 59),
(70, 'Female', 47),
(71, 'Male', 55),
(72, 'Female', 42),
(73, 'Female', 49),
(74, 'Female', 56),
(75, 'Male', 47),
(76, 'Male', 54),
(77, 'Female', 53),
(78, 'Male', 48),
(79, 'Female', 52),
(80, 'Female', 42),
(81, 'Male', 51),
(82, 'Male', 55),
(83, 'Male', 41),
(84, 'Female', 44),
(85, 'Female', 57),
(86, 'Male', 46),
(87, 'Female', 58),
(88, 'Female', 55),
(89, 'Female', 60),
(90, 'Female', 46),
(91, 'Female', 55),
(92, 'Male', 41),
(93, 'Male', 49),
(94, 'Female', 40),
(95, 'Female', 42),
(96, 'Male', 52),
(97, 'Female', 47),
(98, 'Female', 50),
(99, 'Male', 42),
(100, 'Male', 49),
(101, 'Female', 41),
(102, 'Female', 48),
(103, 'Male', 59),
(104, 'Male', 55),
(105, 'Male', 56),
(106, 'Female', 42),
(107, 'Female', 50),
(108, 'Male', 46),
(109, 'Male', 43),
(110, 'Male', 48),
(111, 'Male', 52),
(112, 'Female', 54),
(113, 'Female', 42),
(114, 'Male', 46),
(115, 'Female', 48),
(116, 'Female', 50),
(117, 'Female', 43),
(118, 'Female', 59),
(119, 'Female', 43),
(120, 'Female', 57),
(121, 'Male', 56),
(122, 'Female', 40),
(123, 'Female', 58),
(124, 'Male', 91),
(126, 'Female', 77),
(127, 'Male', 35),
(128, 'Male', 95),
(130, 'Male', 75),
(132, 'Male', 75),
(134, 'Female', 71),
(136, 'Female', 88),
(138, 'Male', 73),
(140, 'Female', 72),
(142, 'Male', 93),
(143, 'Female', 40),
(144, 'Female', 87),
(146, 'Male', 97),
(147, 'Male', 36),
(148, 'Female', 74),
(150, 'Male', 90),
(152, 'Male', 88),
(154, 'Female', 76),
(156, 'Female', 89),
(158, 'Female', 78),
(160, 'Female', 73),
(161, 'Female', 35),
(162, 'Female', 83),
(164, 'Female', 93),
(166, 'Female', 75),
(168, 'Female', 95),
(170, 'Male', 63),
(172, 'Male', 75),
(174, 'Male', 92),
(176, 'Female', 86),
(178, 'Male', 69),
(180, 'Male', 90),
(182, 'Female', 86),
(184, 'Female', 88),
(185, 'Female', 39),
(186, 'Male', 97),
(188, 'Male', 68),
(190, 'Female', 85),
(192, 'Female', 69),
(194, 'Female', 91),
(196, 'Female', 79),
(198, 'Male', 74),
(200, 'Male', 83);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`Roll_no`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
