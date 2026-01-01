class Context:
    def __init__(self, txt="", meta=None):
        self.text = txt
        self.sub_contexts = None
        self.meta = meta or {} # currently only for 'gpt_applied' state

    def to_dict(self):
        return {
            "text": self.text,
            "meta": self.meta,
            "sub_contexts": [c.to_dict() for c in self.sub_contexts] if self.sub_contexts else None
        }

    @staticmethod
    def from_dict(d):
        c = Context(d.get("text", ""), d.get("meta", {}))
        if d.get("sub_contexts"):
            c.sub_contexts = [Context.from_dict(x) for x in d["sub_contexts"]]
        else:
            c.sub_contexts = None
        return c

    def copy(self):
        c = Context(self.text, self.meta.copy())
        if self.sub_contexts:
            c.sub_contexts = [sub.copy() for sub in self.sub_contexts]
        return c
    
    def clone(self):
        return Context.from_dict(self.to_dict())

    def print_tree(self, indent=0):
        pad = "  " * indent
        print(f"{pad}{{")
        print(f"{pad}  text: {self.text!r}")
        if self.meta:
            print(f"{pad}  meta: {self.meta}")
        if self.sub_contexts:
            print(f"{pad}  sub_contexts: [")
            for child in self.sub_contexts:
                child.print_tree(indent + 2)
            print(f"{pad}  ]")
        print(f"{pad}}}")