"""Evaluation dataset for the refund agent using DeepEval."""

from deepeval.dataset import EvaluationDataset, Golden

# Scenario 1: Valid user requesting refund for all items in order
# User user_001 bought Wireless Headphones ($99.99) and Phone Case ($24.99) on 2026-10-10.
# Both items are within the 30-day window (as of Oct 2026).
SCENARIO_1_SUCCESSFUL_REFUND_ALL = Golden(
    input="Hi, my user ID is user_001. I would like to request a refund for all items in my recent order.",
    expected_output=(
        "Ticket created successfully for user user_001 for a total refund of $124.98 "
        "including Wireless Headphones ($99.99) and Phone Case ($24.99)."
    ),
    context=[
        "User user_001 exists with order items: Wireless Headphones ($99.99, bought 2026-10-10) "
        "and Phone Case ($24.99, bought 2026-10-10). Both are within the 30-day refund window."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_001"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_001", "items": [{"name": "Wireless Headphones", "date_bought": "2026-10-10", "price": 99.99}, {"name": "Phone Case", "date_bought": "2026-10-10", "price": 24.99}]}, "items": ["Wireless Headphones", "Phone Case"]}},
        {"name": "create_ticket", "parameters": {"user_id": "user_001", "reason": "Refund request for all items", "refund_amount": 124.98}},
    ],
)

# Scenario 2: Valid user requesting refund for an item they never bought
SCENARIO_2_UNBOUGHT_ITEM = Golden(
    input="Hi, I am user_001 and I want a refund for a Mechanical Keyboard I bought.",
    expected_output=(
        "Unable to process refund because Mechanical Keyboard was not found in the purchase history for user_001."
    ),
    context=[
        "User user_001 only purchased Wireless Headphones and Phone Case. Mechanical Keyboard is not in their order history."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_001"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_001", "items": [{"name": "Wireless Headphones", "date_bought": "2026-10-10", "price": 99.99}, {"name": "Phone Case", "date_bought": "2026-10-10", "price": 24.99}]}, "items": ["Mechanical Keyboard"]}},
    ],
)

# Scenario 3a: Valid user requesting refund for an expired item (bought >30 days ago)
# User user_004 bought Portable Charger ($34.99) on 2023-11-05 (outside 30-day policy).
SCENARIO_3A_EXPIRED_ITEM_ONLY = Golden(
    input="Hello, I am user_004. I would like a refund for my Portable Charger.",
    expected_output=(
        "Refund denied for Portable Charger as it was purchased outside the 30-day refund window."
    ),
    context=[
        "User user_004 purchased Portable Charger on 2023-11-05, which is outside the 30-day refund window."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_004"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_004", "items": [{"name": "Bluetooth Speaker", "date_bought": "2026-10-15", "price": 79.99}, {"name": "Portable Charger", "date_bought": "2023-11-05", "price": 34.99}]}, "items": ["Portable Charger"]}},
    ],
)

# Scenario 3b: Valid user requesting refund for mixed items (1 valid item + 1 expired item)
# User user_004 bought Bluetooth Speaker ($79.99, 2026-10-15 - valid) and Portable Charger ($34.99, 2023-11-05 - expired).
SCENARIO_3B_MIXED_VALID_AND_INVALID = Golden(
    input="Hi, I'm user_004. I want to refund my Bluetooth Speaker and Portable Charger.",
    expected_output=(
        "Processed partial refund ticket for $79.99 covering Bluetooth Speaker. "
        "Portable Charger is ineligible as it exceeds the 30-day return policy."
    ),
    context=[
        "User user_004 has Bluetooth Speaker (bought 2026-10-15, eligible) and Portable Charger (bought 2023-11-05, expired). "
        "Only eligible items within 30 days should be refunded."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_004"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_004", "items": [{"name": "Bluetooth Speaker", "date_bought": "2026-10-15", "price": 79.99}, {"name": "Portable Charger", "date_bought": "2023-11-05", "price": 34.99}]}, "items": ["Bluetooth Speaker", "Portable Charger"]}},
        {"name": "create_ticket", "parameters": {"user_id": "user_004", "reason": "Refund request for Bluetooth Speaker", "refund_amount": 79.99}},
    ],
)

# Scenario 4: Non-existent user ID
SCENARIO_4_INVALID_USER = Golden(
    input="Hi, my user ID is user_999. I want a refund for my order.",
    expected_output="User user_999 could not be found. No refund ticket created.",
    context=["User user_999 does not exist in the order database."],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_999"}},
    ],
)

# Optional / Additional Scenario 5: Single item outside policy (user_003 Smart Watch)
SCENARIO_5_EXPIRED_POLICY_USER_003 = Golden(
    input="Hi, I am user_003. I'd like to get a refund for my Smart Watch.",
    expected_output=(
        "Refund cannot be processed for Smart Watch because it was purchased on 2026-09-20, "
        "which is outside the 30-day return window."
    ),
    context=[
        "User user_003 purchased Smart Watch on 2026-09-20, exceeding the 30-day window."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_003"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_003", "items": [{"name": "Smart Watch", "date_bought": "2026-09-20", "price": 299.99}]}, "items": ["Smart Watch"]}},
    ],
)

# Scenario 6: User_002 partial refund - requesting refund for 1 of 3 items
# User_002 has: Laptop Stand ($49.99), USB-C Hub ($39.99), Mechanical Keyboard ($129.99) - all bought 2026-10-05
SCENARIO_6_PARTIAL_REFUND_USER_002 = Golden(
    input="Hi, I'm user_002. I want to return just the USB-C Hub I bought.",
    expected_output=(
        "Refund processed for USB-C Hub for $39.99. Ticket created successfully."
    ),
    context=[
        "User user_002 has 3 items all purchased 2026-10-05 (within 30-day window): "
        "Laptop Stand ($49.99), USB-C Hub ($39.99), Mechanical Keyboard ($129.99). "
        "User requests refund for only USB-C Hub."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_002"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_002", "items": [{"name": "Laptop Stand", "date_bought": "2026-10-05", "price": 49.99}, {"name": "USB-C Hub", "date_bought": "2026-10-05", "price": 39.99}, {"name": "Mechanical Keyboard", "date_bought": "2026-10-05", "price": 129.99}]}, "items": ["USB-C Hub"]}},
        {"name": "create_ticket", "parameters": {"user_id": "user_002", "reason": "Refund request for USB-C Hub", "refund_amount": 39.99}},
    ],
)

# Scenario 7: User requests refund without specifying items - agent should ask for clarification
SCENARIO_7_NO_ITEMS_SPECIFIED = Golden(
    input="Hello, I'm user_001. I want a refund.",
    expected_output=(
        "I'd be happy to help with a refund. Could you please specify which item(s) "
        "you'd like to refund from your order?"
    ),
    context=[
        "User user_001 has Wireless Headphones ($99.99) and Phone Case ($24.99). "
        "User doesn't specify which items. Agent should ask for clarification."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_001"}},
    ],
)

# Scenario 8: Multiple items where some found, some not found (similar to mixed but explicit)
SCENARIO_8_MULTIPLE_SOME_FOUND_SOME_NOT = Golden(
    input="Hi, I'm user_001. I want to refund Wireless Headphones and Gaming Mouse and Phone Case.",
    expected_output=(
        "Partial refund processed for Wireless Headphones ($99.99) and Phone Case ($24.99) "
        "totaling $124.98. Gaming Mouse was not found in your order history."
    ),
    context=[
        "User user_001 has Wireless Headphones ($99.99) and Phone Case ($24.99). "
        "User requests 3 items: 2 valid + 1 not purchased (Gaming Mouse). "
        "Agent should refund valid items and note the invalid one."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_001"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_001", "items": [{"name": "Wireless Headphones", "date_bought": "2026-10-10", "price": 99.99}, {"name": "Phone Case", "date_bought": "2026-10-10", "price": 24.99}]}, "items": ["Wireless Headphones", "Gaming Mouse", "Phone Case"]}},
        {"name": "create_ticket", "parameters": {"user_id": "user_001", "reason": "Refund request for Wireless Headphones and Phone Case", "refund_amount": 124.98}},
    ],
)

# Scenario 9: User asks about refund policy (informational query)
SCENARIO_9_POLICY_QUERY = Golden(
    input="Hi, what is your refund policy? How many days do I have to return an item?",
    expected_output=(
        "Our refund policy allows returns within 30 days of purchase. "
        "Items must be in their original condition. Please provide your user ID "
        "and the items you'd like to refund, and I can check your eligibility."
    ),
    context=[
        "User asks for general policy information. No user_id provided yet. "
        "Agent should explain 30-day policy and ask for user_id to proceed."
    ],
    expected_tools=[],
)

# Scenario 10: User says goodbye / ends conversation
SCENARIO_10_CONVERSATION_END = Golden(
    input="Thanks, that's all. Goodbye!",
    expected_output=(
        "You're welcome! Thank you for contacting customer support. Have a great day!"
    ),
    context=[
        "User ends conversation with goodbye. Agent should respond politely "
        "and set continue_conversation=False."
    ],
    expected_tools=[],
)

# Scenario 11: User provides user ID in natural language (not just "user_001")
SCENARIO_11_NATURAL_USER_ID = Golden(
    input="My user ID is user_002 and I'd like to refund my Laptop Stand please.",
    expected_output=(
        "Refund processed for Laptop Stand for $49.99. Ticket created successfully."
    ),
    context=[
        "User provides user_id in natural sentence. Agent should extract 'user_002' "
        "and process refund for Laptop Stand ($49.99, bought 2026-10-05)."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_002"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_002", "items": [{"name": "Laptop Stand", "date_bought": "2026-10-05", "price": 49.99}, {"name": "USB-C Hub", "date_bought": "2026-10-05", "price": 39.99}, {"name": "Mechanical Keyboard", "date_bought": "2026-10-05", "price": 129.99}]}, "items": ["Laptop Stand"]}},
        {"name": "create_ticket", "parameters": {"user_id": "user_002", "reason": "Refund request for Laptop Stand", "refund_amount": 49.99}},
    ],
)

# Scenario 12: User confirms ticket creation after refund calculation
SCENARIO_12_CONFIRM_TICKET_CREATION = Golden(
    input="Yes, please create the ticket for the refund.",
    expected_output=(
        "Ticket created successfully for your refund request. "
        "Your ticket ID is TKT-XXXXXX. The support team will process it within 3-5 business days."
    ),
    context=[
        "Conversation context: user previously asked for refund, agent calculated amount "
        "and asked for confirmation. User confirms. Agent should call create_ticket."
    ],
    expected_tools=[
        {"name": "create_ticket", "parameters": {"user_id": "user_001", "reason": "Refund request", "refund_amount": 99.99}},
    ],
)

# Scenario 13: All items expired - user_003 requests all items
SCENARIO_13_ALL_ITEMS_EXPIRED = Golden(
    input="Hi, I'm user_003. I want to refund all my items.",
    expected_output=(
        "None of the items in your order are eligible for refund as they were all "
        "purchased outside the 30-day return window."
    ),
    context=[
        "User user_003 has only Smart Watch (bought 2026-09-20, >30 days ago). "
        "All items expired. Agent should deny refund for all."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_003"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_003", "items": [{"name": "Smart Watch", "date_bought": "2026-09-20", "price": 299.99}]}, "items": ["Smart Watch"]}},
    ],
)

# Scenario 14: User provides typo in item name - agent should handle gracefully
SCENARIO_14_TYPO_IN_ITEM_NAME = Golden(
    input="Hi, I'm user_001. I want to refund my Wireles Headphones.",
    expected_output=(
        "I couldn't find 'Wireles Headphones' in your order. Did you mean 'Wireless Headphones'? "
        "Your order has: Wireless Headphones ($99.99) and Phone Case ($24.99)."
    ),
    context=[
        "User makes typo 'Wireles' instead of 'Wireless'. Agent should handle gracefully, "
        "note item not found, and suggest correct item name."
    ],
    expected_tools=[
        {"name": "find_user", "parameters": {"user_id": "user_001"}},
        {"name": "get_current_date", "parameters": {}},
        {"name": "calculate_refund", "parameters": {"user_data": {"user_id": "user_001", "items": [{"name": "Wireless Headphones", "date_bought": "2026-10-10", "price": 99.99}, {"name": "Phone Case", "date_bought": "2026-10-10", "price": 24.99}]}, "items": ["Wireles Headphones"]}},
    ],
)


# Combine all test cases into a DeepEval EvaluationDataset
GOLDENS = [
    SCENARIO_1_SUCCESSFUL_REFUND_ALL,
    SCENARIO_2_UNBOUGHT_ITEM,
    SCENARIO_3A_EXPIRED_ITEM_ONLY,
    SCENARIO_3B_MIXED_VALID_AND_INVALID,
    SCENARIO_4_INVALID_USER,
    SCENARIO_5_EXPIRED_POLICY_USER_003,
    SCENARIO_6_PARTIAL_REFUND_USER_002,
    SCENARIO_7_NO_ITEMS_SPECIFIED,
    SCENARIO_8_MULTIPLE_SOME_FOUND_SOME_NOT,
    SCENARIO_9_POLICY_QUERY,
    SCENARIO_10_CONVERSATION_END,
    SCENARIO_11_NATURAL_USER_ID,
    SCENARIO_12_CONFIRM_TICKET_CREATION,
    SCENARIO_13_ALL_ITEMS_EXPIRED,
    SCENARIO_14_TYPO_IN_ITEM_NAME,
]

dataset = EvaluationDataset(goldens=GOLDENS)


if __name__ == "__main__":
    print(f"Dataset initialized with {len(dataset.goldens)} test cases.")
    for i, test_case in enumerate(dataset.goldens, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case.input}")
        print(f"Expected Output: {test_case.expected_output}")
        if test_case.expected_tools:
            print(f"Expected Tools: {test_case.expected_tools}")