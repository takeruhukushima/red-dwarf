# Red Dwarf

A dim[ensional] red[uction] library for [stellarpunk][] democracy into the long haul.

> Stars are fires that burn for thousands of years. Some of them burn slow and
> long, like red dwarfs. Others-blue giants-burn their fuel so fast they shine
> across great distances, and are easy to see. As they start to run out of
> fuel, they burn helium, grow even hotter, and explode in a supernova.
> Supernovas, they're brighter than the brightest galaxies. They die, and
> everyone watches them go. -- Jodi Picoult, My Sister's Keeper

> If advanced alien civilizations really are out there, the planets of red
> dwarf stars could be ideal places to find them. -- [How the Universe Works
> (S5E8)](https://youtu.be/3Lq-mI6lgmA?t=375), Discovery

Inspiration: https://chatgpt.com/share/677f7690-7188-800b-85e5-816aaa7cc8f9

## Goals

Code that aspires to embody and support democratic values should be...

- **Legible.** It should be explorable and auditable not just to researchers, but to as many curious citizens as possible.
- **Re-usable.** It should be easily used in contexts in which it's original creators did not anticipate, nor perhaps even desire.
- **Collectively stewarded.** It should grow and govern itself democratically and in a plural fashion.

## Usage

```py
from red_dwarf.polis import PolisClient

client = PolisClient()
# Source: https://github.com/compdemocracy/openData/blob/master/scoop-hivemind.ubi/votes.csv
client.load_data('data/ubi.4yy3sh84js/votes.csv')
# Source: https://pol.is/api/v3/comments?conversation_id=6bkf4ujff9&moderation=true&include_voting_patterns=true
client.load_data('data/ubi.4yy3sh84js/comments.json')
print(client.get_participant_row_mask())
print(client.get_statement_col_mask())
print(client.vote_matrix)
client.run_pca()
client.run_clustering()
```

<!-- Links -->
   [stellarpunk]: https://www.youtube.com/watch?v=opnkQVZrhAw
