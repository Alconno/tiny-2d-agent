from core.logging import get_logger
log = get_logger(__name__) 

class Variable():
    def __init__(self, name=None, type=None, desc=None, value=None):
        self.name = name
        self.type = type
        self.desc = desc
        self.value = value 

    def __repr__(self):
        return f"Variable(name={self.name!r}, type={self.type!r}, desc={self.desc!r}, value={self.value!r})"

    @staticmethod
    def extract_structured_var(ctx_str):
        log.debug("ctx str: ", ctx_str) # name|type|desc
        
        parts = list(map(str.strip, ctx_str.split("|")))
        while len(parts) < 3:
            parts.append(None)
        
        name, typ, desc = parts[:3]
        name = name or ""
        typ = typ or ""
        desc = desc or ""
        return Variable(name=name, type=typ, desc=desc)