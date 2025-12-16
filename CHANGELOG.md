# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [0.1.0] - 2025-12-16

### Added
- **Core Data Architecture**: Plug-in data adapter architecture with abstract `Data` base class
- **Pandas Adapter**: DataFrame-based data handling with CSV I/O, compression support, and rich type inference
- **NumPy Adapter**: High-performance array-based implementation with optimized counting and memory management
- **Oracle Adapter**: Synthetic data generation from Bayesian Networks for algorithm validation
- **Data Randomisation**: Comprehensive randomization capabilities for names, order, and sampling
- **In-Memory Optimization**: Efficient categorical counting using `bincount` and smart memory management
- **Type System**: Support for categorical, and continuous variable types (but not yet mixtures)
- **BNFit Interface**: Full compatibility with causaliq-core Bayesian Network fitting
- **Statistical Operations**: Marginal computation
- **Documentation**: Complete API reference and architectural documentation with MkDocs
