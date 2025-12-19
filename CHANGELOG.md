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

## [0.2.0] - 2025-12-19

### Added
- **Score Functions Module**: Comprehensive scoring framework for Bayesian networks and DAGs
- **Multiple Score Types**: Support for entropy-based (BIC, AIC, log-likelihood), Bayesian (BDE, K2, BDJ, BDS), and Gaussian (BGE, BIC-g, loglik-g) scores
- **Network Evaluation**: Complete DAG and Bayesian Network scoring with per-node breakdowns via `dag_score()` and `bn_score()`
- **Node-Level Scoring**: Independent evaluation of nodes with their parents using `node_score()`
- **Specialized Score Functions**: Individual scoring functions for categorical and Gaussian data
- **Parameter Validation**: Automatic parameter checking and default value assignment via `check_score_params()`
- **Score Documentation**: Complete API documentation for all scoring functions
- **Architecture Integration**: Score functions integrated into overall causaliq-data architecture

### Changed
- Enhanced API documentation structure to include score module
- Updated navigation and overview documentation to cover scoring capabilities

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
