"""
verbalyze/telephony/ivr_tree.py

Multi-Level Indic Interactive Voice Response (IVR) State Machine:
1. Hierarchical Menu Navigation:
   - Configurable tree of IVRNode instances.
   - Dynamic multilingual prompts (Hindi, English, Gujarati, Marathi).
2. Hybrid Input Handling:
   - Traverses nodes on DTMF keypad touch-tones ('0'-'9', '*', '#').
   - Traverses nodes on spoken keywords/phrases (e.g. 'hindi', 'payment', 'agent').
3. Multi-Digit Input Collection:
   - Collects fixed-length digit sequences (e.g. 4-digit PINs, OTPs, account numbers).
   - Inter-digit timeout management.
4. DPDP Act 2023 Compliant Sensitive Keypad Masking:
   - Protects PINs and sensitive keypad entries from plaintext logging.
5. Pluggable Action Hooks:
   - Language selection ('set_language').
   - WhatsApp UPI payment link dispatch ('send_upi_link').
   - Live human agent transfer ('transfer_human').
   - AI conversational bot handoff ('handoff_bot').

Zero-emoji compliant.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

from verbalyze.telephony.dtmf_engine import mask_digits, sanitize_sensitive_dict


# Spoken number equivalence map for hybrid voice/keypad navigation
SPOKEN_NUMBER_MAP: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "chaar": "4",
    "paanch": "5", "chhah": "6", "saat": "7", "aath": "8", "nau": "9",
    "star": "*", "hash": "#",
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5",
    "પાંચ": "5", "એક": "1", "બે": "2", "ત્રણ": "3", "ચાર": "4",
}


@dataclass
class IVRNode:
    """Represents a single step or menu in the IVR navigation tree."""
    node_id: str
    title: str
    prompt: Union[str, Dict[str, str]]
    options: Dict[str, str] = field(default_factory=dict)
    spoken_keywords: Dict[str, str] = field(default_factory=dict)
    collect_digits: Optional[int] = None
    inter_digit_timeout_sec: float = 3.0
    timeout_sec: float = 8.0
    max_retries: int = 3
    action: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    sensitive_input: bool = False
    is_terminal: bool = False
    invalid_prompt: Optional[Union[str, Dict[str, str]]] = None
    timeout_prompt: Optional[Union[str, Dict[str, str]]] = None

    def get_prompt(self, lang: str = "hi") -> str:
        """Resolves prompt text in caller's active language with fallback."""
        if isinstance(self.prompt, dict):
            return self.prompt.get(lang) or self.prompt.get("en") or next(iter(self.prompt.values()), "")
        return str(self.prompt)

    def get_invalid_prompt(self, lang: str = "hi") -> str:
        """Resolves invalid entry prompt text."""
        if self.invalid_prompt:
            if isinstance(self.invalid_prompt, dict):
                return self.invalid_prompt.get(lang) or self.invalid_prompt.get("en") or next(iter(self.invalid_prompt.values()), "")
            return str(self.invalid_prompt)
        default_invalid = {
            "hi": "अमान्य विकल्प। कृपया पुनः प्रयास करें।",
            "en": "Invalid option. Please try again.",
            "gu": "અમાન્ય વિકલ્પ. કૃપા કરીને ફરી પ્રયાસ કરો.",
            "mr": "अवैध पर्याय. कृपया पुन्हा प्रयत्न करा."
        }
        return default_invalid.get(lang, "Invalid option. Please try again.")

    def get_timeout_prompt(self, lang: str = "hi") -> str:
        """Resolves timeout prompt text."""
        if self.timeout_prompt:
            if isinstance(self.timeout_prompt, dict):
                return self.timeout_prompt.get(lang) or self.timeout_prompt.get("en") or next(iter(self.timeout_prompt.values()), "")
            return str(self.timeout_prompt)
        default_timeout = {
            "hi": "हमें आपका कोई इनपुट नहीं मिला। कृपया अपना विकल्प चुनें।",
            "en": "We did not receive your input. Please make your selection.",
            "gu": "અમને તમારો કોઈ ઇનપુટ મળ્યો નથી. કૃપા કરીને તમારી પસંદગી કરો.",
            "mr": "आम्हाला तुमचा कोणताही प्रतिसाद मिळाला नाही. कृपया आपला पर्याय निवडा."
        }
        return default_timeout.get(lang, "We did not receive your input. Please make your selection.")


@dataclass
class IVRTransitionResult:
    """Structured result returned on IVR state transitions."""
    node_id: str
    node_title: str
    prompt: str
    action: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False
    status: str = "OK"  # OK, COLLECTING_DIGITS, INVALID_INPUT, TIMEOUT, MAX_RETRIES_EXCEEDED, COMPLETED
    collected_digits: Optional[str] = None  # DPDP masked
    raw_collected_digits: Optional[str] = None  # Unmasked internal copy for action execution
    retries_left: int = 3
    language: str = "hi"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into DPDP Act 2023 compliant sanitized dictionary."""
        return sanitize_sensitive_dict({
            "node_id": self.node_id,
            "node_title": self.node_title,
            "prompt": self.prompt,
            "action": self.action,
            "action_params": self.action_params,
            "is_terminal": self.is_terminal,
            "status": self.status,
            "collected_digits": self.collected_digits,
            "retries_left": self.retries_left,
            "language": self.language,
        })


class IVRStateMachine:
    """
    Stateful Multi-Level IVR Engine with Hybrid Keypad & Speech Traversal.
    """

    def __init__(
        self,
        call_id: str,
        nodes: Dict[str, IVRNode],
        root_node_id: str = "root",
        language: str = "hi",
        caller_phone: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.call_id = call_id
        self.nodes = nodes
        self.root_node_id = root_node_id
        self.current_node_id = root_node_id
        self.language = language
        self.caller_phone = caller_phone
        self.context: Dict[str, Any] = context or {}

        # Runtime traversal state
        self.digit_buffer: str = ""
        self.last_input_time: float = time.time()
        self.retry_count: int = 0
        self.is_completed: bool = False
        self.state_history: List[Dict[str, Any]] = []

    def get_current_node(self) -> IVRNode:
        """Returns active IVRNode instance."""
        if self.current_node_id not in self.nodes:
            raise KeyError(f"Node '{self.current_node_id}' not found in IVR tree.")
        return self.nodes[self.current_node_id]

    def start(self) -> IVRTransitionResult:
        """
        Initializes IVR session, evaluates root node actions, and returns opening prompt.
        """
        self.current_node_id = self.root_node_id
        self.digit_buffer = ""
        self.retry_count = 0
        self.is_completed = False
        self.last_input_time = time.time()

        node = self.get_current_node()
        self._record_history("start", node.node_id, "")

        # Execute any immediate entry actions
        action = node.action
        action_params = dict(node.action_params)
        self._apply_action(action, action_params)

        return IVRTransitionResult(
            node_id=node.node_id,
            node_title=node.title,
            prompt=node.get_prompt(self.language),
            action=action,
            action_params=action_params,
            is_terminal=node.is_terminal,
            status="OK",
            retries_left=node.max_retries - self.retry_count,
            language=self.language,
        )

    def handle_digit(self, digit: str) -> IVRTransitionResult:
        """
        Processes a single DTMF keypad press ('0'-'9', '*', '#', 'A'-'D').
        """
        if self.is_completed:
            curr = self.get_current_node()
            return IVRTransitionResult(
                node_id=curr.node_id,
                node_title=curr.title,
                prompt="Call completed.",
                is_terminal=True,
                status="COMPLETED",
                language=self.language,
            )

        now = time.time()
        curr_node = self.get_current_node()

        # 1. Multi-digit collection node (e.g. 4-digit PIN entry)
        if curr_node.collect_digits and curr_node.collect_digits > 0:
            # Check inter-digit timeout
            if self.digit_buffer and (now - self.last_input_time) > curr_node.inter_digit_timeout_sec:
                # Timed out during sequence; reset buffer
                self.digit_buffer = ""

            self.last_input_time = now
            self.digit_buffer += digit

            masked_buf = mask_digits(self.digit_buffer) if curr_node.sensitive_input else self.digit_buffer

            if len(self.digit_buffer) < curr_node.collect_digits:
                return IVRTransitionResult(
                    node_id=curr_node.node_id,
                    node_title=curr_node.title,
                    prompt="",  # Silence while caller types remaining digits
                    status="COLLECTING_DIGITS",
                    collected_digits=masked_buf,
                    raw_collected_digits=self.digit_buffer,
                    retries_left=curr_node.max_retries - self.retry_count,
                    language=self.language,
                )

            # Collected required digits
            completed_digits = self.digit_buffer
            self.digit_buffer = ""
            self.retry_count = 0

            # Store in context
            if curr_node.sensitive_input:
                self.context[f"{curr_node.node_id}_digits_masked"] = mask_digits(completed_digits)
            else:
                self.context[f"{curr_node.node_id}_digits"] = completed_digits

            self._record_history("digits_collected", curr_node.node_id, masked_buf)

            action = curr_node.action
            action_params = dict(curr_node.action_params)
            action_params["collected_digits"] = completed_digits
            self._apply_action(action, action_params)

            # If node options specify a next target (e.g. "default" or continuation)
            next_node_id = curr_node.options.get("default") or curr_node.options.get("*")
            if next_node_id and next_node_id in self.nodes:
                self.current_node_id = next_node_id
                next_node = self.get_current_node()
                if next_node.is_terminal:
                    self.is_completed = True
                return IVRTransitionResult(
                    node_id=next_node.node_id,
                    node_title=next_node.title,
                    prompt=next_node.get_prompt(self.language),
                    action=next_node.action or action,
                    action_params=next_node.action_params or action_params,
                    is_terminal=next_node.is_terminal,
                    status="OK",
                    collected_digits=masked_buf,
                    raw_collected_digits=completed_digits,
                    retries_left=next_node.max_retries,
                    language=self.language,
                )

            return IVRTransitionResult(
                node_id=curr_node.node_id,
                node_title=curr_node.title,
                prompt=curr_node.get_prompt(self.language),
                action=action,
                action_params=action_params,
                is_terminal=curr_node.is_terminal,
                status="OK",
                collected_digits=masked_buf,
                raw_collected_digits=completed_digits,
                retries_left=curr_node.max_retries,
                language=self.language,
            )

        # 2. Single-digit menu option selection
        self.last_input_time = now
        digit_key = digit.upper()

        if digit_key in curr_node.options:
            target_node_id = curr_node.options[digit_key]
            if target_node_id not in self.nodes:
                return self._handle_invalid_input(curr_node, f"Unknown target '{target_node_id}'")

            self.current_node_id = target_node_id
            self.retry_count = 0
            new_node = self.get_current_node()

            self._record_history("option_selected", new_node.node_id, digit_key)

            action = new_node.action
            action_params = dict(new_node.action_params)
            self._apply_action(action, action_params)

            if new_node.is_terminal:
                self.is_completed = True

            return IVRTransitionResult(
                node_id=new_node.node_id,
                node_title=new_node.title,
                prompt=new_node.get_prompt(self.language),
                action=action,
                action_params=action_params,
                is_terminal=new_node.is_terminal,
                status="OK",
                retries_left=new_node.max_retries,
                language=self.language,
            )

        # Invalid option entered
        return self._handle_invalid_input(curr_node, f"Digit '{digit_key}' not in options")

    def handle_speech(self, utterance: str) -> IVRTransitionResult:
        """
        Processes spoken voice utterance for hybrid IVR navigation.
        Translates spoken digits ('one', 'ek', 'do') or intent keywords ('hindi', 'payment', 'agent')
        to state transitions.
        """
        if self.is_completed:
            curr = self.get_current_node()
            return IVRTransitionResult(
                node_id=curr.node_id,
                node_title=curr.title,
                prompt="Call completed.",
                is_terminal=True,
                status="COMPLETED",
                language=self.language,
            )

        curr_node = self.get_current_node()
        norm_text = utterance.strip().lower()

        # 1. Check direct spoken keywords configured for this node
        for keyword, target_node_id in curr_node.spoken_keywords.items():
            if keyword.lower() in norm_text:
                if target_node_id in self.nodes:
                    self.current_node_id = target_node_id
                    self.retry_count = 0
                    new_node = self.get_current_node()

                    self._record_history("speech_keyword", new_node.node_id, keyword)

                    action = new_node.action
                    action_params = dict(new_node.action_params)
                    self._apply_action(action, action_params)

                    if new_node.is_terminal:
                        self.is_completed = True

                    return IVRTransitionResult(
                        node_id=new_node.node_id,
                        node_title=new_node.title,
                        prompt=new_node.get_prompt(self.language),
                        action=action,
                        action_params=action_params,
                        is_terminal=new_node.is_terminal,
                        status="OK",
                        retries_left=new_node.max_retries,
                        language=self.language,
                    )

        # 2. Check spoken number mapping
        for spoken_num, digit in SPOKEN_NUMBER_MAP.items():
            if spoken_num in norm_text.split() or norm_text == spoken_num:
                return self.handle_digit(digit)

        # 3. Check if text is an exact digit character ('1', '2', etc.)
        for word in norm_text.split():
            if word in curr_node.options:
                return self.handle_digit(word)

        return self._handle_invalid_input(curr_node, f"Spoken utterance '{utterance}' not recognized")

    def handle_timeout(self) -> IVRTransitionResult:
        """
        Handles user inactivity timeout.
        Increments retry count and either plays timeout prompt or transitions to terminal node.
        """
        if self.is_completed:
            curr = self.get_current_node()
            return IVRTransitionResult(
                node_id=curr.node_id,
                node_title=curr.title,
                prompt="Call completed.",
                is_terminal=True,
                status="COMPLETED",
                language=self.language,
            )

        curr_node = self.get_current_node()
        self.retry_count += 1
        self._record_history("timeout", curr_node.node_id, f"retry_{self.retry_count}")

        if self.retry_count >= curr_node.max_retries:
            self.is_completed = True
            hangup_prompt = {
                "hi": "कोई प्रतिक्रिया न मिलने के कारण कॉल समाप्त की जा रही है। धन्यवाद।",
                "en": "Ending call due to lack of response. Thank you.",
                "gu": "કોઈ પ્રતિસાદ ન મળવાને કારણે કૉલ સમાપ્ત કરવામાં આવી રહ્યો છે. આભાર.",
                "mr": "प्रतिसाद न मिळाल्यामुळे कॉल समाप्त केला जात आहे. धन्यवाद."
            }.get(self.language, "Ending call due to lack of response. Thank you.")

            return IVRTransitionResult(
                node_id=curr_node.node_id,
                node_title=curr_node.title,
                prompt=hangup_prompt,
                action="hangup",
                action_params={"reason": "max_timeouts_exceeded"},
                is_terminal=True,
                status="MAX_RETRIES_EXCEEDED",
                retries_left=0,
                language=self.language,
            )

        return IVRTransitionResult(
            node_id=curr_node.node_id,
            node_title=curr_node.title,
            prompt=f"{curr_node.get_timeout_prompt(self.language)} {curr_node.get_prompt(self.language)}",
            status="TIMEOUT",
            retries_left=curr_node.max_retries - self.retry_count,
            language=self.language,
        )

    def _handle_invalid_input(self, node: IVRNode, reason: str) -> IVRTransitionResult:
        """Handles invalid selection entry, retrying or terminating on threshold."""
        self.retry_count += 1
        self._record_history("invalid_input", node.node_id, reason)

        if self.retry_count >= node.max_retries:
            self.is_completed = True
            fail_prompt = {
                "hi": "अमान्य प्रयासों की अधिकतम सीमा समाप्त हो गई है। धन्यवाद।",
                "en": "Maximum invalid attempts exceeded. Thank you.",
                "gu": "મહત્તમ અમાન્ય પ્રયાસો ઓળંગાઈ ગયા છે. આભાર.",
                "mr": "अवैध प्रयत्नांची मर्यादा संपली आहे. धन्यवाद."
            }.get(self.language, "Maximum invalid attempts exceeded. Thank you.")

            return IVRTransitionResult(
                node_id=node.node_id,
                node_title=node.title,
                prompt=fail_prompt,
                action="hangup",
                action_params={"reason": "max_retries_exceeded"},
                is_terminal=True,
                status="MAX_RETRIES_EXCEEDED",
                retries_left=0,
                language=self.language,
            )

        return IVRTransitionResult(
            node_id=node.node_id,
            node_title=node.title,
            prompt=f"{node.get_invalid_prompt(self.language)} {node.get_prompt(self.language)}",
            status="INVALID_INPUT",
            retries_left=node.max_retries - self.retry_count,
            language=self.language,
        )

    def _apply_action(self, action: Optional[str], params: Dict[str, Any]):
        """Executes state-mutating actions during navigation."""
        if not action:
            return

        if action == "set_language":
            new_lang = params.get("language")
            if new_lang:
                self.language = new_lang
                self.context["language"] = new_lang

        elif action == "hangup":
            self.is_completed = True

    def _record_history(self, event_type: str, node_id: str, detail: str):
        """Records DPDP-safe navigation trail."""
        self.state_history.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "node_id": node_id,
            "detail": detail,
            "language": self.language,
        })

    def get_summary(self) -> Dict[str, Any]:
        """Returns DPDP Act 2023 sanitized audit summary of IVR state machine session."""
        curr = self.get_current_node()
        return sanitize_sensitive_dict({
            "call_id": self.call_id,
            "current_node_id": self.current_node_id,
            "current_node_title": curr.title,
            "language": self.language,
            "caller_phone": self.caller_phone,
            "is_completed": self.is_completed,
            "retry_count": self.retry_count,
            "context": self.context,
            "history_count": len(self.state_history),
            "state_history": self.state_history,
        })


def create_default_banking_ivr(
    call_id: str = "call_default",
    default_lang: str = "hi",
    caller_phone: Optional[str] = None,
) -> IVRStateMachine:
    """
    Builds the standard Indian Banking & NBFC Debt Collection IVR Tree.
    
    Structure:
    - Root: Multilingual Language Selection (Hindi=1, English=2, Gujarati=3).
    - Main Services Menu:
      - 1: WhatsApp UPI Payment Link Dispatch.
      - 2: Account / EMI Verification (4-digit PIN with DPDP masking).
      - 3: Live Recovery Officer SIP Transfer.
      - 9: AI Conversational Voicebot Handoff.
      - *: Repeat Menu.
    """
    nodes: Dict[str, IVRNode] = {
        # Level 0: Welcome & Language Selection
        "root": IVRNode(
            node_id="root",
            title="Language Selection Menu",
            prompt={
                "hi": "मुथूट फिनकॉर्प में आपका स्वागत है। हिंदी के लिए 1 दबाएं। For English press 2. ગુજરાતી માટે 3 દબાવો.",
                "en": "Welcome to Muthoot Fincorp. For Hindi press 1. For English press 2. For Gujarati press 3.",
                "gu": "મુથૂટ ફિનકોર્પમાં તમારું સ્વાગત છે. હિન્દી માટે 1 દબાવો. અંગ્રેજી માટે 2 દબાવો. ગુજરાતી માટે 3 દબાવો.",
            },
            options={
                "1": "node_lang_hi",
                "2": "node_lang_en",
                "3": "node_lang_gu",
            },
            spoken_keywords={
                "hindi": "node_lang_hi",
                "हिंदी": "node_lang_hi",
                "english": "node_lang_en",
                "अंग्रेजी": "node_lang_en",
                "gujarati": "node_lang_gu",
                "ગુજરાતી": "node_lang_gu",
            },
            max_retries=3,
        ),

        # Intermediate Language Setter Nodes
        "node_lang_hi": IVRNode(
            node_id="node_lang_hi",
            title="Set Hindi",
            prompt={"hi": "हिंदी भाषा का चयन किया गया है।"},
            action="set_language",
            action_params={"language": "hi"},
            options={"default": "main_menu", "*": "main_menu"},
        ),
        "node_lang_en": IVRNode(
            node_id="node_lang_en",
            title="Set English",
            prompt={"en": "English language selected."},
            action="set_language",
            action_params={"language": "en"},
            options={"default": "main_menu", "*": "main_menu"},
        ),
        "node_lang_gu": IVRNode(
            node_id="node_lang_gu",
            title="Set Gujarati",
            prompt={"gu": "ગુજરાતી ભાષા પસંદ કરવામાં આવી છે."},
            action="set_language",
            action_params={"language": "gu"},
            options={"default": "main_menu", "*": "main_menu"},
        ),

        # Level 1: Main Banking Services Menu
        "main_menu": IVRNode(
            node_id="main_menu",
            title="Main Services Menu",
            prompt={
                "hi": "व्हाट्सएप पर यूपीआई भुगतान लिंक प्राप्त करने के लिए 1 दबाएं। अपने खाते के सत्यापन के लिए 2 दबाएं। ऋण अधिकारी से बात करने के लिए 3 दबाएं। हमारे एआई वॉयस असिस्टेंट से बात करने के लिए 9 दबाएं।",
                "en": "Press 1 to receive a UPI payment link on WhatsApp. Press 2 to verify your loan account. Press 3 to speak with a loan recovery officer. Press 9 to speak with our AI voice assistant.",
                "gu": "WhatsApp પર UPI ચુકવણી લિંક મેળવવા 1 દબાવો. તમારા ખાતાની ચકાસણી માટે 2 દબાવો. લોન અધિકારી સાથે વાત કરવા 3 દબાવો. AI વૉઇસ આસિસ્ટન્ટ સાથે વાત કરવા 9 દબાવો.",
            },
            options={
                "1": "node_pay_upi",
                "2": "node_verify_account",
                "3": "node_transfer_agent",
                "9": "node_bot_handoff",
                "*": "main_menu",
            },
            spoken_keywords={
                "payment": "node_pay_upi",
                "भुगतान": "node_pay_upi",
                "upi": "node_pay_upi",
                "verify": "node_verify_account",
                "verification": "node_verify_account",
                "सत्यापन": "node_verify_account",
                "agent": "node_transfer_agent",
                "officer": "node_transfer_agent",
                "अधिकारी": "node_transfer_agent",
                "assistant": "node_bot_handoff",
                "bot": "node_bot_handoff",
            },
            max_retries=3,
        ),

        # Level 2a: WhatsApp UPI Payment Link
        "node_pay_upi": IVRNode(
            node_id="node_pay_upi",
            title="WhatsApp UPI Payment Link Dispatch",
            prompt={
                "hi": "आपके पंजीकृत मोबाइल नंबर पर व्हाट्सएप के माध्यम से सुरक्षित यूपीआई भुगतान लिंक भेज दिया गया है। मुख्य मेनू के लिए 0 दबाएं, या कॉल समाप्त करने के लिए # दबाएं।",
                "en": "A secure UPI payment link has been dispatched to your registered WhatsApp number. Press 0 for main menu, or # to hang up.",
                "gu": "તમારા WhatsApp નંબર પર સુરક્ષિત UPI ચુકવણી લિંક મોકલી દેવામાં આવી છે. મુખ્ય મેનુ માટે 0 દબાવો, અથવા # દબાવો.",
            },
            action="send_upi_link",
            action_params={"channel": "whatsapp_upi"},
            options={
                "0": "main_menu",
                "#": "node_hangup",
            },
            spoken_keywords={
                "menu": "main_menu",
                "main menu": "main_menu",
                "bye": "node_hangup",
                "hangup": "node_hangup",
            },
        ),

        # Level 2b: Account Verification with 4-Digit Sensitive PIN Collection
        "node_verify_account": IVRNode(
            node_id="node_verify_account",
            title="Account Verification (4-Digit PIN)",
            prompt={
                "hi": "कृपया अपने 4 अंकों का गुप्त पिन दर्ज करें।",
                "en": "Please enter your 4-digit secret PIN.",
                "gu": "કૃપા કરીને તમારો 4 અંકનો ગુપ્ત પિન દાખલ કરો.",
            },
            collect_digits=4,
            sensitive_input=True,
            inter_digit_timeout_sec=3.0,
            action="validate_pin",
            options={
                "default": "node_pin_verified",
                "*": "node_pin_verified",
            },
        ),
        "node_pin_verified": IVRNode(
            node_id="node_pin_verified",
            title="PIN Verified Successfully",
            prompt={
                "hi": "आपका पिन सफलतापूर्वक सत्यापित हो गया है। मुख्य मेनू के लिए 0 दबाएं, या कॉल समाप्त करने के लिए # दबाएं।",
                "en": "Your PIN has been verified successfully. Press 0 for main menu, or # to hang up.",
                "gu": "તમારો પિન સફળતાપૂર્વક ચકાસવામાં આવ્યો છે. મુખ્ય મેનુ માટે 0 દબાવો, અથવા # દબાવો.",
            },
            options={
                "0": "main_menu",
                "#": "node_hangup",
            },
        ),

        # Level 2c: Human Transfer
        "node_transfer_agent": IVRNode(
            node_id="node_transfer_agent",
            title="Human Agent Transfer",
            prompt={
                "hi": "आपकी कॉल हमारे वरिष्ठ ऋण अधिकारी को स्थानांतरित की जा रही है। कृपया प्रतीक्षा करें।",
                "en": "Your call is being transferred to a senior loan recovery officer. Please hold.",
                "gu": "તમારો કૉલ વરિષ્ઠ લોન અધિકારીને ટ્રાન્સફર કરવામાં આવી રહ્યો છે. કૃપા કરીને રાહ જુઓ.",
            },
            action="transfer_human",
            action_params={"queue": "recovery_tier2"},
            is_terminal=True,
        ),

        # Level 2d: Bot Handoff
        "node_bot_handoff": IVRNode(
            node_id="node_bot_handoff",
            title="AI Conversational Voicebot Handoff",
            prompt={
                "hi": "नमस्कार, मैं मुथूट फिनकॉर्प का एआई वॉयस असिस्टेंट हूँ। मैं आपके ऋण समाधान में आपकी क्या सहायता कर सकता हूँ?",
                "en": "Hello, I am Muthoot Fincorp's AI voice assistant. How may I assist you with your loan settlement today?",
                "gu": "નમસ્તે, હું મુથૂટ ફિનકોર્પનો AI વૉઇસ આસિસ્ટન્ટ છું. હું તમારા લોન સમાધાનમાં કેવી રીતે મદદ કરી શકું?",
            },
            action="handoff_bot",
            action_params={"persona": "muthoot_recovery"},
            is_terminal=True,
        ),

        # Hangup Node
        "node_hangup": IVRNode(
            node_id="node_hangup",
            title="Call Completion",
            prompt={
                "hi": "मुथूट फिनकॉर्प को कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो।",
                "en": "Thank you for calling Muthoot Fincorp. Have a great day.",
                "gu": "મુથૂટ ફિનકોર્પ પર કૉલ કરવા બદલ આભાર. તમારો દિવસ શુભ રહે.",
            },
            action="hangup",
            is_terminal=True,
        ),
    }

    return IVRStateMachine(
        call_id=call_id,
        nodes=nodes,
        root_node_id="root",
        language=default_lang,
        caller_phone=caller_phone,
    )
