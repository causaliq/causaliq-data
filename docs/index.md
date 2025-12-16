# 🔢  CausalIQ Data

<!-- add in coverage and CI badges when repo is public -->

![Python Versions](https://img.shields.io/badge/python-3.9%2C%203.10%2C%203.11%2C%203.12-blue)

## Welcome

Welcome to the documentation for **CausalIQ Data** — part of the [CausalIQ ecosystem](https://causaliq.org) for intelligent causal discovery. 

The CausalIQ Data project provides the data-related capabilities that causal
discovery requires. 

---

## Overview

**CausalIQ Data** provides:

 -  ⚡ **data import and caching** - data can be imported from standard 
tabular formats (comma-separated variables) and cached for high performance
 - 🎯 **graph scoring** - provide graph score derived from the data which is
 the objective function used by *score-based* structure learning algorithms. This is 
 based upon how likely the data is to be seen for a given graph, typically
 modified by a penalty for complex graphs (e.g. BIC score), or modified
 by a prior belief about the graph strcuture (e.g. BDeu score)
 - 🔗 **independence tests** - used to determine conditional independence tests
 which are intrinsic to the operataion of *constraint-based* structure
 learning algorithms.

This site provides detailed documentation, including: development roadmap, user guide, architectural vision, design notes, and API reference for users and contributors.

---

## Quickstart & Installation

For a quickstart guide and installation instructions, see the [README on GitHub](https://github.com/causaliq/causaliq-data#readme).

---

## Documentation Contents

- [Development Roadmap](roadmap.md): roadmap of upcoming features
- [User Guide](userguide/introduction.md): comprehensive user guide
- [Architecture](architecture/overview.md): overall architecture and design notes
- [API Reference](api/overview.md): complete reference for Python code
- [Development Guidelines](https://github.com/causaliq/causaliq-data/blob/main/CONTRIBUTING.md): CausalIQ guidelines for developers
- [Changelog](https://github.com/causaliq/causaliq-data/blob/main/CHANGELOG.md)
- [License](https://github.com/causaliq/causaliq-data/blob/main/LICENSE)

---

## Support & Community

- [GitHub Issues](https://github.com/causaliq/causaliq-data/issues): Report bugs or request features.
- [GitHub Discussions](https://github.com/causaliq/causaliq-data/discussions): Ask questions and join the community.

---

**Tip:**  
Use the navigation sidebar to explore the documentation.  
For the latest code and releases, visit the [causaliq-data GitHub repository](https://github.com/causaliq/causaliq-data).

---

**Supported Python Versions**: 3.9, 3.10, 3.11, 3.12  
**Default Python Version**: 3.11