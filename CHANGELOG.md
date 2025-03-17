# Changelog

## [Unreleased][] (YYYY-MM-DD)
### Fixed
- Relax seaborn version constraint to be compatible with TabPFN. ([#16](https://github.com/polis-community/red-dwarf/issues/16))

### Changes
- Implement `utils.calculate_representativeness()` function. ([#22](https://github.com/polis-community/red-dwarf/issues/22))
- Add color legend for labels in `data_presenter.generate_figure()`. [`d55f535`](https://github.com/polis-community/red-dwarf/pull/24/commits/d55f53588de72620abb984d7c1ac27f8a31d5478) ([#22](https://github.com/polis-community/red-dwarf/issues/22))

## [0.1.1][] (2025-03-04)
### Bugfixes
- Fix publishing issue with missing license file. ([#19](https://github.com/polis-community/red-dwarf/issues/19))
    - Workaround for [`pypa/setuptools#4769`](https://github.com/pypa/setuptools/issues/4759).
- Change package name from `reddwarf` to `red-dwarf-democracy`.

## [0.1.0][] (2025-03-04)

- Add Agora/ZKorum to README as sponsor.
- Add low-level stateless functions in `reddwarf.utils`.
    - Add first-pass unit test coverage, and CI.
- Add mid-level stateless `reddwarf.agora` implementation.
    - Add preferred types/modules and function definitions.
- Add code coverage reports for unit tests.
- Document library usage in Jupyter notebooks.
- Create documentation website using `mkdocs`.
- Add `make` targets for common development tasks, and default help target.
- Experimental
    - Add high-level stateful Polis client class implementation.
    - Add high-level stateful data loader class.
        - Support loading from JSON API, CSV export API, local files, and remote directory.
        - Support rate-limiting.
        - Support request caching.
        - Support bypass of Cloudflare for all API endpoints.
    - Add high-level stateful data presenter class.
    - Add integration tests.

<!-- Links -->
   [Unreleased]: https://github.com/polis-community/red-dwarf/compare/v0.1.1...main
   [0.1.1]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.1
   [0.1.0]: https://github.com/polis-community/red-dwarf/releases/tag/v0.1.0
