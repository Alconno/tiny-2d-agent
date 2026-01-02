

import re

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100, "thousand": 1000
}

def text_to_number(text):
    tokens = text.lower().split()
    total = 0
    current = 0
    for t in tokens:
        if t in WORDS:
            val = WORDS[t]
            if val == 100 or val == 1000:
                current *= val
            else:
                current += val
        else:
            if current:
                total += current
                current = 0
    return total + current

def parse_delay(text):
    text = text.lower().strip()
    
    numeric = re.findall(r"\d+(?:\.\d+)?", text)
    if numeric:
        value = float(numeric[0])

        if "sec" in text or "s " in text:
            return int(value * 1000)
        return int(value) 
    
    num_spelled = text_to_number(text)
    if num_spelled > 0:
        if "sec" in text or "s " in text:
            return num_spelled * 1000
        return num_spelled
    
    return None





def parse_sign_number(expr: str):
    # Parses chained numeric conditions like '>10<50' or '>=5<=20'.
    # Returns list of (sign, number) tuples.
    expr = expr.strip()
    pattern = r"(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, expr)
    
    # Case: no explicit sign, just a number like "15"
    if not matches:
        expr = expr.lstrip("=<>") # remove garbage 
        return [("==", float(expr))]
    
    return [(sign, float(num)) for sign, num in matches]
