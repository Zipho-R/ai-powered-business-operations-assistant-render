import re

try:
    from AITicketClassifier.predict import predict_ticket
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False


# ----------------------------
# Categories & Keywords
# ----------------------------
categories = {
    "IT": [
        "laptop", "pc", "desktop", "computer", "server", "network", "wifi", "internet",
        "router", "password", "login", "system", "software", "application", "keyboard",
        "app", "hardware", "email", "printer", "scanner", "database", "update", "screen",
        "backup", "cloud", "security", "virus", "firewall", "bug", "crash", "crashed",
        "slow", "connection", "phone", "data", "device", "breach", "log", "access",
        "authentication", "mfa", "otp", "vpn", "website", "portal", "file",
        "folder", "download", "upload", "malware", "phishing", "locked", "frozen",
        "blue screen", "error", "outage", "downtime", "email account"
    ],
    "HR": [
        "leave", "vacation", "sick", "salary", "pay", "benefits", "bonus", "payslip",
        "promotion", "hiring", "recruitment", "termination", "appraisal", "training",
        "policy", "contract", "discipline", "performance", "overtime", "attendance",
        "shift", "resignation", "onboarding", "exit", "employee", "wage", "payroll",
        "evaluation", "feedback", "absence", "insurance", "resign", "discrimination",
        "conflict", "harassment", "grievance", "misconduct", "manager", "supervisor",
        "wellness", "workplace", "staff", "timesheet", "personal", "medical certificate",
        "maternity", "paternity", "probation", "bullying"
    ],
    "Finance": [
        "invoice", "payment", "budget", "expense", "reimbursement", "tax", "profit",
        "loss", "accounting", "financial", "billing", "credit", "debit", "balance",
        "transaction", "financial report", "audit", "funds", "cash", "cost", "funding",
        "ledger", "statement", "forecast", "purchase", "receivable", "payable", "loan",
        "investment", "fraud", "fraudulent", "charges", "refund", "receipt", "quote",
        "quotation", "supplier", "vendor", "bank", "unauthorized",
        "unauthorised", "overcharged", "duplicate", "claim", "allowance", "petty cash",
        "purchase order", "po", "vat", "bank account", "financial account"
    ],
    "Operations": [
        "delivery", "delayed", "schedule", "logistics", "shipment", "transport", "route",
        "warehouse", "inventory", "stock", "order", "supply", "production",
        "maintenance", "process", "quality", "inspection", "planning", "coordination",
        "tracking", "dispatch", "assembly", "management", "equipment", "task",
        "resource", "workflow", "project", "timing", "deadline", "shift", "accident",
        "injury", "facility", "cleaning", "repair", "machine", "vehicle", "courier",
        "supplier", "site", "floor", "building", "safety", "incident", "capacity",
        "materials", "shortage", "delay"
    ]
}


# ----------------------------
# Priority Keywords
# ----------------------------
IT_high = [
    "down", "crash", "crashed", "hacked", "failure", "breach", "critical error",
    "virus", "malware", "ransomware", "outage", "lost data", "data loss",
    "corrupt", "critical", "security incident", "phishing", "cannot work",
    "system down", "server down", "network down", "locked out", "urgent access"
]

IT_medium = [
    "slow", "cannot login", "login", "issue", "bug", "not working", "access",
    "problem", "update", "printer", "email issue", "password reset", "vpn",
    "software install", "screen", "keyboard", "mouse", "application error",
    "minor error", "intermittent", "connection issue", "bad connection"
]

HR_high = [
    "harassment", "discrimination", "abuse", "legal", "termination", "misconduct",
    "grievance", "workplace safety", "threat", "bullying", "victimisation",
    "unfair dismissal", "confidential complaint", "salary not paid", "missing salary",
    "pay not received"
]

HR_medium = [
    "conflict", "salary query", "salary", "leave", "payslip", "policy", "request",
    "overtime", "benefits", "evaluation", "appraisal", "contract", "training",
    "attendance", "shift change", "onboarding", "probation", "performance review",
    "medical certificate"
]

Finance_high = [
    "fraud", "fraudulent", "failed payment", "overdue", "missing funds",
    "unauthorized", "unauthorised", "duplicate payment", "critical", "audit issue",
    "payment failed", "account blocked", "incorrect debit", "large discrepancy",
    "suspicious transaction", "overcharged", "tax penalty", "compliance issue"
]

Finance_medium = [
    "invoice", "payment", "reimbursement", "expense", "billing", "budget",
    "request", "query", "receipt", "refund", "quote", "quotation", "supplier",
    "vendor", "purchase order", "po", "claim", "allowance", "petty cash",
    "statement", "balance", "cost centre"
]

Operations_high = [
    "failed", "accident", "injury", "stopped", "broken", "stockout", "urgent",
    "critical", "safety incident", "machine down", "production stopped",
    "delivery failed", "major delay", "shortage", "no stock", "site incident",
    "equipment failure", "blocked workflow"
]

Operations_medium = [
    "delay", "delayed", "schedule", "inventory", "tracking", "shipment", "logistics",
    "planning", "coordination", "update", "maintenance", "repair", "quality issue",
    "inspection", "resource", "workflow", "deadline", "courier", "dispatch",
    "supplier delay", "process issue"
]

urgent_words = [
    "today", "urgent", "urgently", "asap", "immediately", "now", "right away",
    "same day", "emergency", "critical", "critically", "cannot continue",
    "blocked", "before end of day"
]


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(ticket):
    text = str(ticket).lower()
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = text.replace("’", "'").replace("‘", "'")

    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "hasn't": "has not",
        "haven't": "have not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "i'm": "i am",
        "it's": "it is"
    }

    for short, full in contractions.items():
        text = text.replace(short, full)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_keyword(text, keyword):
    keyword = keyword.lower().strip()

    if " " in keyword:
        return keyword in text

    return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None


weights = {
    "HR": {
        "salary": 2,
        "payslip": 3,
        "leave": 2,
        "harassment": 4,
        "discrimination": 4,
        "grievance": 3,
        "termination": 3,
        "bullying": 4
    },
    "Finance": {
        "salary": 1,
        "payment": 3,
        "invoice": 3,
        "reimbursement": 3,
        "fraud": 5,
        "fraudulent": 5,
        "unauthorized": 4,
        "unauthorised": 4,
        "audit": 3
    },
    "IT": {
        "breach": 5,
        "virus": 4,
        "malware": 4,
        "hacked": 5,
        "password": 2,
        "login": 2,
        "wifi": 2
    },
    "Operations": {
        "accident": 4,
        "injury": 4,
        "stockout": 4,
        "delivery": 2,
        "inventory": 2,
        "safety": 3,
        "delayed": 2
    }
}


# ----------------------------
# Rule-Based Category Backup
# ----------------------------
def classify_ticket(ticket):
    text = normalize_text(ticket)
    scores = {cat: 0 for cat in categories}

    for cat, keywords in categories.items():
        for keyword in keywords:
            if contains_keyword(text, keyword):
                scores[cat] += weights.get(cat, {}).get(keyword, 1)

    matched = {cat: score for cat, score in scores.items() if score > 0}

    if not matched:
        return ["Unclassified"], []

    max_score = max(matched.values())

    primary = [cat for cat, score in matched.items() if score == max_score]
    secondary = [cat for cat, score in matched.items() if 0 < score < max_score]

    return primary, secondary


# ----------------------------
# Hybrid ML + Rules
# ----------------------------
def classify_ticket_hybrid(ticket):
    rule_primary, rule_secondary = classify_ticket(ticket)

    empty_result = {
        "primary": rule_primary,
        "secondary": rule_secondary,
        "category_confidence": 0,
        "priority_confidence": 0,
        "urgency_confidence": 0,
        "ml_priority": None,
        "ml_urgency": None,
        "method": "Rule-Based"
    }

    if not ML_AVAILABLE:
        return empty_result

    try:
        ml_result = predict_ticket(ticket)
    except Exception:
        return empty_result

    ml_category = ml_result.get("category")
    category_confidence = ml_result.get("category_confidence", 0)

    result_base = {
        "category_confidence": category_confidence,
        "priority_confidence": ml_result.get("priority_confidence", 0),
        "urgency_confidence": ml_result.get("urgency_confidence", 0),
        "ml_priority": ml_result.get("priority"),
        "ml_urgency": ml_result.get("urgency")
    }

    # Strong ML confidence
    if category_confidence >= 60:
        secondary = [cat for cat in rule_primary + rule_secondary if cat != ml_category]

        return {
            "primary": [ml_category],
            "secondary": secondary,
            "method": "Machine Learning",
            **result_base
        }

    # Medium confidence + rules agree
    if category_confidence >= 40 and ml_category in rule_primary:
        return {
            "primary": rule_primary,
            "secondary": rule_secondary,
            "method": "Hybrid ML + Rules",
            **result_base
        }

    # Low confidence → fallback to rules
    if category_confidence < 40:

        # Rules found a category
        if rule_primary != ["Unclassified"]:
            return {
                "primary": rule_primary,
                "secondary": rule_secondary,
                "method": "Rule-Based Backup",
                **result_base
            }

        # Rules ALSO failed
        return {
            "primary": ["Unclassified"],
            "secondary": [],
            "method": "Human Review Required",
            **result_base
        }

    # Default fallback
    return {
        "primary": [ml_category],
        "secondary": rule_secondary,
        "method": "Hybrid ML + Rules",
        **result_base
    }

# ----------------------------
# Priority / Urgency Backup
# ----------------------------
priority_rules = {
    "IT": (IT_high, IT_medium),
    "HR": (HR_high, HR_medium),
    "Finance": (Finance_high, Finance_medium),
    "Operations": (Operations_high, Operations_medium)
}


def get_priority(ticket, category):
    text = normalize_text(ticket)
    high_keywords, medium_keywords = priority_rules.get(category, ([], []))

    if any(contains_keyword(text, word) for word in high_keywords):
        return "High", 3

    if any(contains_keyword(text, word) for word in medium_keywords):
        return "Medium", 2

    return "Low", 1


def get_urgency(ticket):
    text = normalize_text(ticket)

    if any(contains_keyword(text, word) for word in urgent_words):
        return "Immediate"

    return "Not Immediate"


def get_urgency_code(label):
    return 1 if label == "Immediate" else 2


def priority_to_number(priority):
    mapping = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }
    return mapping.get(priority, 1)


# ----------------------------
# Risk / Approval / Routing
# ----------------------------
def evaluate_ai_risk(ticket, category):
    text = normalize_text(ticket)
    risks = []

    if category == "Unclassified":
        risks.append("Low confidence classification")

    if category == "HR" and any(word in text for word in [
        "harassment", "discrimination", "abuse", "termination", "grievance", "bullying"
    ]):
        risks.append("Sensitive HR issue")

    if category == "Finance" and any(word in text for word in [
        "fraud", "fraudulent", "unauthorized", "unauthorised", "missing",
        "suspicious", "duplicate payment", "overcharged"
    ]):
        risks.append("Financial risk")

    if category == "IT" and any(word in text for word in [
        "breach", "virus", "hacked", "malware", "ransomware", "phishing"
    ]):
        risks.append("Cybersecurity risk")

    if category == "Operations" and any(word in text for word in [
        "accident", "injury", "safety incident", "equipment failure"
    ]):
        risks.append("Operational safety risk")

    if not risks:
        return "No major risk detected"

    return ", ".join(sorted(set(risks)))


def requires_approval(category, ticket):
    text = normalize_text(ticket)

    approval_words = [
        "fraud", "fraudulent", "termination", "legal", "breach", "hacked",
        "harassment", "discrimination", "injury", "accident",
        "unauthorized", "unauthorised", "suspicious", "missing funds",
        "duplicate payment", "overcharged"
    ]

    return any(contains_keyword(text, word) for word in approval_words)

def get_workflow_status(category, ticket):
    if category == "Unclassified":
        return "Pending Review", "No", "Not Required"

    if requires_approval(category, ticket):
        return "Pending Approval", "Yes", "Pending"

    return "Open", "No", "Not Required"


def get_routing_suggestion(category):
    routing_map = {
        "IT": "IT Department",
        "HR": "HR Department",
        "Finance": "Finance Department",
        "Operations": "Operations Department"
    }

    return routing_map.get(category, "Unclassified Queue")


def get_transparency_note():
    return (
        "This ticket was classified using a hybrid AI system combining machine learning "
        "and rule-based automation. Risk flags, sensitive cases, finance approvals, "
        "and low-confidence tickets may require human review."
    )

# ----------------------------
# Response Generation
# ----------------------------
def get_response(category, priority_num, urgency_num):
    messages = {
        "IT": {
            3: "This is a critical IT issue requiring immediate technical attention. The IT team will prioritise system stability, security, and user access restoration." if urgency_num == 1 else "This is a high-priority IT issue. The IT team will prioritise it and work toward restoring normal service as soon as possible.",
            2: "Your IT request has been marked as urgent and will be attended to by the IT team today." if urgency_num == 1 else "Your IT request has been logged and scheduled. The IT team will resolve it according to support priority.",
            1: "Your IT request has been received and will be addressed when support resources are available." if urgency_num == 2 else "Your IT request has been received and will be reviewed as soon as possible because it may affect your work today."
        },
        "HR": {
            3: "This is a sensitive or high-priority HR matter requiring careful attention. It will be escalated for appropriate human review and handling." if urgency_num == 1 else "This is a high-priority HR matter and will be handled carefully through the appropriate HR process.",
            2: "Your HR request has been marked as urgent and will be prioritised by HR today." if urgency_num == 1 else "Your HR request has been received and will be processed by HR in due course.",
            1: "Your HR request has been logged and will be handled as part of standard HR processes." if urgency_num == 2 else "Your HR request has been received and will be attended to shortly."
        },
        "Finance": {
            3: "This is a critical financial issue and may require approval or audit review before further processing. The Finance team will prioritise it." if urgency_num == 1 else "This is a high-priority financial matter and will be prioritised for review and resolution.",
            2: "Your finance request has been marked as urgent and will be handled by the Finance team today." if urgency_num == 1 else "Your finance request has been scheduled and will be processed by the Finance team soon.",
            1: "Your finance request has been logged and will be processed when resources are available." if urgency_num == 2 else "Your finance request has been received and will be handled as soon as possible."
        },
        "Operations": {
            3: "This is a critical operations issue requiring immediate review. The Operations team will prioritise safety, continuity, and workflow impact." if urgency_num == 1 else "This is a high-priority operations matter and is being actively reviewed.",
            2: "Your operations request has been prioritised and will be handled by the Operations team today." if urgency_num == 1 else "Your operations request has been scheduled and will be handled by Operations soon.",
            1: "Your operations request has been logged and will be handled in the normal workflow." if urgency_num == 2 else "Your operations request has been received and will be attended to shortly."
        },
        "Unclassified": {
            1: "Your request has been received. It will be reviewed manually and assigned to the appropriate department.",
            2: "Your request has been received. It will be reviewed manually and assigned to the appropriate department.",
            3: "Your request has been received. It appears sensitive or unclear and will be reviewed manually before routing."
        }
    }

    return messages.get(category, messages["Unclassified"]).get(
        priority_num,
        "Your request has been received and will be reviewed."
    )

# ----------------------------
# Full Processing
# ----------------------------
def process_ticket(ticket):
    clean_text = normalize_text(ticket)

    hybrid = classify_ticket_hybrid(ticket)

    primary_categories = hybrid["primary"]
    secondary_categories = hybrid["secondary"]
    all_categories = primary_categories + secondary_categories

    main_category = primary_categories[0]

    category_confidence = hybrid["category_confidence"]
    priority_confidence = hybrid["priority_confidence"]
    urgency_confidence = hybrid["urgency_confidence"]
    classification_method = hybrid["method"]

    ml_priority = hybrid["ml_priority"]
    ml_urgency = hybrid["ml_urgency"]

    rule_priority, rule_priority_num = get_priority(ticket, main_category)
    rule_urgency = get_urgency(ticket)

    if ml_priority and priority_confidence >= 55:
        overall_priority_label = ml_priority
        priority_method = "Machine Learning"
    else:
        overall_priority_label = rule_priority
        priority_method = "Rule-Based Backup"

    if ml_urgency and urgency_confidence >= 55:
        urgency_label = ml_urgency
        urgency_method = "Machine Learning"
    else:
        urgency_label = rule_urgency
        urgency_method = "Rule-Based Backup"

    overall_priority_num = priority_to_number(overall_priority_label)
    urgency_num = get_urgency_code(urgency_label)

    if len(all_categories) > 1:
        response = "All relevant departments will review your request and respond through the appropriate workflow."
    else:
        response = get_response(main_category, overall_priority_num, urgency_num)

    all_categories = list(dict.fromkeys(all_categories))
    csv_categories = ",".join(all_categories)

    main_category = all_categories[0]
    related_categories = all_categories[1:]

    if related_categories:
        display_categories = f"{main_category} | Also related: {', '.join(related_categories)}"

    csv_categories = ",".join(all_categories)

    if secondary_categories:
        display_categories = f"{main_category} | Also related: {', '.join(secondary_categories)}"
    else:
        display_categories = main_category

    status, approval_required, approval_status = get_workflow_status(main_category, ticket)

    return {
        "clean_text": clean_text,
        "categories": csv_categories,
        "category_display": display_categories,
        "priority": overall_priority_label,
        "priority_code": f"{overall_priority_num}.{urgency_num}",
        "urgency": urgency_label,
        "response": response,
        "risk_flags": evaluate_ai_risk(ticket, main_category),
        "transparency_note": get_transparency_note(),
        "routed_to": get_routing_suggestion(main_category),
        "status": status,
        "approval_required": approval_required,
        "approval_status": approval_status,

        # Existing fields kept for database safety
        "ml_confidence": category_confidence,
        "classification_method": classification_method,

        # New Phase 2 fields
        "category_confidence": category_confidence,
        "priority_confidence": priority_confidence,
        "urgency_confidence": urgency_confidence,
        "priority_method": priority_method,
        "urgency_method": urgency_method
    }


# ----------------------------
# Optional Console Testing Only
# ----------------------------
if __name__ == "__main__":
    while True:
        ticket = input("Enter ticket (or type 'quit'): ").strip()

        if ticket.lower() == "quit":
            break

        if not ticket:
            print("Please enter a valid ticket.\n")
            continue

        result = process_ticket(ticket)

        print("\nTicket:", ticket)
        print("Category:", result["category_display"])
        print("Category Confidence:", result["category_confidence"])
        print("Classification Method:", result["classification_method"])
        print("Priority:", result["priority"], result["priority_code"])
        print("Priority Confidence:", result["priority_confidence"])
        print("Priority Method:", result["priority_method"])
        print("Urgency:", result["urgency"])
        print("Urgency Confidence:", result["urgency_confidence"])
        print("Urgency Method:", result["urgency_method"])
        print("Response:", result["response"])
        print("Risk Flags:", result["risk_flags"])
        print("Routed To:", result["routed_to"])
        print("Status:", result["status"])
        print("Approval Required:", result["approval_required"])
        print("Approval Status:", result["approval_status"])
        print("Transparency:", result["transparency_note"])
        print()