def embd_events(embd_func, possible_events):
    event_map = [
        (phrase, action)
        for aliases, action in possible_events.items()
        for phrase in aliases
    ]

    phrases = [p for p, _ in event_map]
    embeds = embd_func(phrases)

    return [(phrase, emb, action) for (phrase, action), emb in zip(event_map, embeds)]