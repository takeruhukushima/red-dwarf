import pandas as pd
from reddwarf.models import ModeratedEnum

def process_statements(
    statement_data: list[dict] = [],
) -> tuple[pd.DataFrame, list, list, list]:
    """
    Process raw statement data into a dataframe, and various lists of participant IDs.

    Args:
        statement_data (list[dict]): raw list of statement data dicts

    Returns:
        statements_df (pd.DataFrame): Dataframe of statements
        mod_in_statement_ids (list): List of statement IDs to moderate in
        mod_out_statement_ids (list): List of statement IDs to moderate out
        meta_statement_ids (list): List of meta statement IDs

    """
    mod_in_statement_ids = []
    mod_out_statement_ids = []
    meta_statement_ids = []

    statements_df = (pd.DataFrame
        # TODO: See if both "moderated" and "mod" can end up in here. BUG?
        .from_records(statement_data)
        .set_index('statement_id')
        .sort_index()
    )

    for i, row in statements_df.iterrows():
        # TODO: Why does is_meta make this mod-in.
        # Maybe I don't understand what mod-in does...
        # Note: mod-in/mod-out doesn't seem to be actually used in the front-end, so a bug here wouldn't matter.
        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L825-L842
        # TODO: Why doesn't is_strict_moderation matter here?
        if row['is_meta'] or row['moderated'] == ModeratedEnum.APPROVED:
            mod_in_statement_ids.append(i)

        if row['is_meta'] or row['moderated'] == ModeratedEnum.REJECTED:
            mod_out_statement_ids.append(i)

        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L843-L850
        if row['is_meta']:
            meta_statement_ids.append(i)

    return statements_df, mod_in_statement_ids, mod_out_statement_ids, meta_statement_ids