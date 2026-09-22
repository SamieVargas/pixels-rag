"""The constants every other module reads: the model, the fields a question
can filter on, the four routes, and the two JSON Schemas the API is asked to
honour. One source, so the router prompt, the schemas and the filter code
cannot drift apart."""

MODEL = "claude-haiku-4-5-20251001"
CONTRACTS = ("native", "prompt")
KINDS = ("semantic", "filter", "aggregate", "unanswerable")
STATS = ("mean", "median", "count", "min", "max")
OPS = ("<", "<=", ">", ">=", "==", "!=", "contains")

# Field name -> (type, description). Names are the normalized day-record keys
# in core/days.py and the metadata keys on every chunk.
FIELDS = {
    "rating": ("number", "the day's 1 to 5 rating"),
    "sleep_score": ("number", "sleep score 0 to 100 (0 means not logged)"),
    "sleep_hours": ("number", "hours slept"),
    "body_battery": ("number", "Garmin body battery at end of day"),
    "hrv": ("number", "overnight HRV in ms"),
    "rhr": ("number", "resting heart rate"),
    "steps": ("number", "step count"),
    "water": ("number", "water in oz"),
    "garmin_stress": ("number", "Garmin stress score"),
    "mood": ("text", "mood word, e.g. Calm, Anxious, Low"),
    "energy": ("text", "Low, Moderate or High"),
    "anxiety": ("text", "None, Mild, Moderate or Severe"),
    "stress": ("text", "Low, Moderate or High"),
    "brain_fog": ("text", "None, Mild, Moderate or Severe"),
    "sleep_quality": ("text", "Poor, Fair, Good or Excellent"),
    "exercise": ("text", "exercise done, comma separated, e.g. Hot yoga, Walking, Trail running"),
    "activity": ("text", "Low, Moderate or High"),
    "symptoms": ("text", "physical symptoms, e.g. Migraine, Fatigue"),
    "hashimotos": ("text", "Hashimoto's symptoms when logged"),
    "diet": ("text", "Neutral, Good or Poor"),
    "caffeine": ("text", "None, 1 cup, 2 cups, 3+ cups"),
    "social": ("text", "Neutral, Positive, Draining, Very positive"),
    "weather": ("text", "Sunny, Cloudy, Rain, Hot, Storm"),
    "productivity": ("text", "Low, Moderate or High"),
    "weekday": ("text", "Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
    "weekend": ("boolean", "Saturday or Sunday"),
    "yoga": ("boolean", "did yoga"),
    "meditation": ("boolean", "meditated"),
    "journaling": ("boolean", "journaled"),
    "reading": ("boolean", "read"),
    "gratitude": ("boolean", "gratitude practice"),
    "sunlight": ("boolean", "got sunlight"),
    "alcohol": ("boolean", "had alcohol"),
    "fast_food": ("boolean", "had fast food"),
    "late_meals": ("boolean", "ate late"),
    "positive_event": ("boolean", "a positive life event was active"),
    "negative_event": ("boolean", "a negative life event was active"),
}
NUMERIC_FIELDS = tuple(k for k, (t, _) in FIELDS.items() if t == "number")
TEXT_FIELDS = tuple(k for k, (t, _) in FIELDS.items() if t == "text")
BOOLEAN_FIELDS = tuple(k for k, (t, _) in FIELDS.items() if t == "boolean")

# What is never logged, so the router can say so instead of retrieving.
NOT_LOGGED = "blood pressure, weight, meals or food eaten, meetings, who was spoken to, locations, spending amounts, medication doses"


def field_lines():
    return "\n".join(f"- {name} ({typ}): {desc}" for name, (typ, desc) in FIELDS.items())


def nullable_enum(values):
    """A closed set or null. The API's schema validator refuses an enum on a
    field whose type is a list (it reported: enum value 'rating' does not match
    declared type ['string', 'null']), so the two cases are spelled out as
    anyOf, which structured outputs support."""
    return {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}


def router_schema():
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "date_range": {
                "type": "object",
                "properties": {"start": nullable_str, "end": nullable_str},
                "required": ["start", "end"],
                "additionalProperties": False,
            },
            "date_phrase": nullable_str,
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": list(FIELDS)},
                        "op": {"type": "string", "enum": list(OPS)},
                        "value": {"type": ["number", "string", "boolean"]},
                    },
                    "required": ["field", "op", "value"],
                    "additionalProperties": False,
                },
            },
            "aggregate": {
                "type": "object",
                "properties": {
                    "metric": nullable_enum(NUMERIC_FIELDS),
                    "stat": nullable_enum(STATS),
                    "group_by": nullable_str,
                },
                "required": ["metric", "stat", "group_by"],
                "additionalProperties": False,
            },
            "rewritten_query": nullable_str,
        },
        "required": ["kind", "date_range", "date_phrase", "filters", "aggregate", "rewritten_query"],
        "additionalProperties": False,
    }


def answer_schema():
    dates = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "dates": dates},
                    "required": ["text", "dates"],
                    "additionalProperties": False,
                },
            },
            "cited_dates": dates,
            "unanswerable": {"type": "boolean"},
            "why_unanswerable": {"type": ["string", "null"]},
        },
        "required": ["answer", "claims", "cited_dates", "unanswerable", "why_unanswerable"],
        "additionalProperties": False,
    }
