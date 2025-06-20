# Red Dwarf: A Pol.is-inspired library

[![PyPI - Version](https://img.shields.io/pypi/v/red-dwarf)][pypi]
[![Run Tests](https://github.com/polis-community/red-dwarf-democracy/actions/workflows/test.yml/badge.svg)](https://github.com/polis-community/red-dwarf-democracy/actions/workflows/test.yml)

A <em>DIM</em>ensional <em>RED</em>uction library for [stellarpunk][] democracy into the long haul.

> Red dwarf stars are dim red stars. They are hard to see, but are the most common
> type -- the quiet 70% majority. They burn slow, using fuel efficiently,
> making them the longest-living stars in the universe. They'll be around until
> the last light, even supporting habitable planets for billions of years.
> If there's persistant life in the universe, it's likely around a red dwarf.

## Roadmap

For now, see [this related issue](https://github.com/patcon/red-dwarf/issues/4)

## Goals

Code that aspires to embody and support democratic values should be...

- **Legible.** It should be explorable and auditable not just to researchers, but to as many curious citizens as possible.
- **Re-usable.** It should be easily used in contexts in which its original creators did not anticipate, nor perhaps even desire.
- **Collectively stewarded.** It should grow and govern itself democratically and in a plural fashion.

## Sponsors

<p>
  <a href="https://agoracitizen.network" rel="noopener sponsored" target="_blank"><img width="167" src="https://agoracitizen.network/images/big_logo_agora.png" alt="Agora Citizen Network" title="Where citizens converge to exchange and debate ideas" loading="lazy" /></a>
</p>

Red Dwarf is generously sponsored by [ZKorum SAS](https://zkorum.com), creators of the [Agora Citizen Network](https://agoracitizen.network).

Are you or your organization eager to see more platforms and community built around democracy-supporting algorithms like these? **Please consider [getting in touch on Discord](#get-involved) and supporting our continued work!** (ping @patcon)

## Usage

See [`docs/notebooks/polis-implementation-demo.ipynb`][notebook] or [`docs/notebooks/`][notebooks] for other examples.


| screenshot of library-generated notebook | screenshot of Polis-generated report |
|---|---|
| [![screen of the sample jupyter notebook](docs/notebook-screenshot.png)][notebook] | ![screenshot of the polis report](https://imgur.com/blkIEtW.png) |

- [`docs/notebooks/loading-data.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/loading-data.ipynb)
- [`docs/notebooks/heatmap.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/heatmap.ipynb)
- [`docs/notebooks/polis-implementation-demo.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/polis-implementation-demo.ipynb)
- [`docs/notebooks/dump-downloaded-polis-data.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/dump-downloaded-polis-data.ipynb)
- Advanced
   - [`docs/notebooks/map-xids.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/map-xids.ipynb)
   - [`docs/notebooks/experiment-pca-pacmap-localmap-comparison.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/experiment-pca-pacmap-localmap-comparison.ipynb)
   - [`docs/notebooks/strip-pass-votes.ipynb`](https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/strip-pass-votes.ipynb)

## Architecture

This is the generalized pipeline of Polis-like processes that we're aiming to accomodate. (See [issue #53](https://github.com/polis-community/red-dwarf/issues/53#issuecomment-2942923628) for details.)

![](https://github.com/user-attachments/assets/8b7b5bfc-7127-4a27-8316-7528209d7c8e)

## Get Involved

- [Join][pug-discord] the _Polis User Group (PUG)_ **Discord** server.
- Open a **GitHub issue**.
- Submit a **GitHub pull request**.
- Review the [**_Awesome Polis_ directory**][awesome-polis] to learn about related projects, academic papers, and other resources.
   - Use the _"People"_ section to find other individuals and groups working in the field.


## Changelog

See [`CHANGELOG.md`][changelog].

## License

[MPL 2.0: Mozilla Public License 2.0][mplv2] (See [`LICENSE`][license])

<!-- Links -->
   [pypi]: https://pypi.org/project/red-dwarf/
   [stellarpunk]: https://www.youtube.com/watch?v=opnkQVZrhAw
   [notebook]: https://github.com/polis-community/red-dwarf/blob/main/docs/notebooks/polis-implementation-demo.ipynb
   [notebooks]: https://github.com/polis-community/red-dwarf/tree/main/docs/notebooks/
   [ZKorum]: https://github.com/zkorum
   [agora]: https://agoracitizen.network/
   [ngi-funding]: https://trustchain.ngi.eu/zkorum/
   [MPLv2]: https://choosealicense.com/licenses/mpl-2.0/
   [license]: https://github.com/polis-community/red-dwarf/blob/main/LICENSE
   [pug-discord]: https://discord.com/invite/wFWB8kzQpP
   [awesome-polis]: http://patcon.github.io/awesome-polis/
   [changelog]: CHANGELOG.md
