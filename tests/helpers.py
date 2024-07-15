import docdeid as dd


def link_tokens(tokens: list[dd.Token]):
    for token, next_token in zip(tokens, tokens[1:]):
        token.set_next_token(next_token)
        next_token.set_previous_token(token)

    return tokens


def linked_tokens(words: list[str]) -> list[dd.Token]:
    tokens = []
    cidx = 0
    for word in words:
        tokens.append(dd.Token(word, cidx, cidx + len(word)))
        cidx += len(word) + 1

    return link_tokens(tokens)
