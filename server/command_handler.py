def process_command(text):
    text = text.lower().strip()

    # Remove common punctuation
    text = text.replace(".", "")
    text = text.replace(",", "")

    # -------------------------
    # LIGHT ON
    # -------------------------
    if (
        "turn on the light" in text
        or "turn on light" in text
        or text == "the light"
        or text == "light"
    ):
        return {
            "command": "LIGHT_ON",
            "action": "Light turned ON"
        }

    # -------------------------
    # LIGHT OFF
    # -------------------------
    if (
        "turn off the light" in text
        or "turn off light" in text
    ):
        return {
            "command": "LIGHT_OFF",
            "action": "Light turned OFF"
        }

    # -------------------------
    # FAN ON
    # -------------------------
    if (
        "start fan" in text
        or "turn on fan" in text
        or "turn on the fan" in text
    ):
        return {
            "command": "FAN_ON",
            "action": "Fan turned ON"
        }

    # -------------------------
    # FAN OFF
    # -------------------------
    if (
        "stop fan" in text
        or "turn off fan" in text
        or "turn off the fan" in text
    ):
        return {
            "command": "FAN_OFF",
            "action": "Fan turned OFF"
        }

    # -------------------------
    # DOOR
    # -------------------------
    if "open door" in text:
        return {
            "command": "DOOR_OPEN",
            "action": "Door OPEN command received"
        }

    # -------------------------
    # UNKNOWN
    # -------------------------
    return {
        "command": "UNKNOWN",
        "action": "No matching command"
    }


if __name__ == "__main__":

    tests = [
        "turn on the light",
        "turn on light",
        "the light",
        "light",
        "turn off the light",
        "start fan",
        "turn on the fan",
        "stop fan",
        "open door",
        "hello"
    ]

    for text in tests:
        print(
            text,
            "->",
            process_command(text)
        )