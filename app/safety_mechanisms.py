import re
from typing import Set, Tuple, List
from langchain_together import Together
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from app.chat.Conversation import Conversation
from app.chat.Message import Message
nltk.download('punkt_tab')
nltk.download('stopwords')

from agents.TherapyAgent import TherapyAgent

class SafetyMechanisms:
    def __init__(self, therapy_agent: TherapyAgent, escalation_threshold: float = 0.8):
        self.therapy_agent = therapy_agent
        self.escalation_threshold = escalation_threshold
        self.together_model = Together(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0.3, max_tokens=512)

        self.explicit_content_patterns: Set[str] = {
            r'\b(sex|porn|explicit|nude|naked)\b',
            r'\b(fuck|shit|damn|bitch)\b',
            r'\b(sex(ual)?|porn(ography)?|explicit|nude|naked|nsfw)\b',
            r'\b(f[u\*@]ck|sh[i\*@]t|b[i\*@]tch|d[a\*@]mn|ass(?:hole)?)\b',
            r'\b(dick|cock|penis|vagina|pussy|boob|breast|anal|cum|semen)\b',
            r'\b(masturbat(e|ion)|orgasm|erotic|arousal|foreplay)\b',
            r'\b(hook[ -]?up|one[ -]?night[ -]?stand|friends?[ -]?with[ -]?benefits)\b',
            r'\b(gore|blood|brutal|mutilat(e|ion)|dismember)\b',
            r'\b(torture|maim|slaughter|massacre|execute|murder)\b',
            r'\b(graphic|disturbing|nsfl|trigger)\b'
        }
        
        self.hate_speech_indicators: List[Tuple[str, float]] = [
            ("hate", 0.9),
            ("kill", 1.0),
            ("die", 0.8),
            ("racist", 0.9),
            ("terrorist", 0.8),
            ("vermin", 0.4),
            ("parasite", 0.3),
            ("scum", 0.4),
            ("filth", 0.3),
            ("infidel", 0.8),
            ("heretic", 0.7),
            ("kafir", 0.8),
            ("goyim", 0.7),
            ("exterminate", 0.9),
            ("eliminate", 0.7),
            ("cleanse", 0.8),
            ("purge", 0.8),
        ]
        
        # Based on research papers and crisis hotline trigger words
        self.self_harm_phrases: List[str] = [
            "i can't handle this",
            "i'm so done",
            "i feel numb",
            "i can't keep going",
            "i feel worthless",
            "i hate myself",
            "i just want it to end",
            "i need to escape",
            "want to die",
            "kill myself",
            "end it all",
            "no reason to live",
            "better off dead",
            "cant go on",
            "cant take it anymore",
            "hurt myself",
            "hurting myself",
            "suicide",
            "end my life"
        ]
        
        self.self_harm_keywords: Set[str] = {
            "suicide", "suicidal", "kill", "die", "death", "hurt", 
            "pain", "cutting", "cut", "overdose", "pills", "escape",
            "numb", "hate", "worth", "alone", "despair", "hurting"
        }

    def process_message(self, message: Message, conversation: Conversation) -> Tuple[bool, bool]:
        """
        Process a message and return flags indicating if the message should be escalated and if the conversation
        should be terminated.
        
        Returns:
            Tuple[bool, bool]: (is_escalated, is_terminated)
        """
        # Apply multi-stage content filtering
        is_escalated, is_terminated = self._apply_content_filtering(message, conversation)
        
        return is_escalated, is_terminated

    def _apply_content_filtering(self, message: Message, conversation: Conversation) -> Tuple[bool, bool]:
        """
        Apply multi-stage content filtering and return flags for escalation and termination.
        """
        is_escalated = False
        is_terminated = False

        # Explicit content detection
        if self._contains_explicit_content(message.text):
            is_escalated = True

        # Hate speech identification
        if self._contains_hate_speech(message.text):
            is_escalated = True

        # Mention of self-harm or suicidal ideation
        if self._contains_self_harm_or_suicide(message.text):
            is_escalated = True
            is_terminated = True

        # Escalation protocol
        if is_escalated and self._should_escalate(conversation):
            # TODO: figure out what escalation looks like 
            return is_escalated, is_terminated

        return is_escalated, is_terminated

    def _should_escalate(self, conversation: Conversation) -> bool:
        """
        Determine if the conversation should be escalated based on the escalation threshold.
        """
        # Calculate the escalation score based on the conversation history
        escalation_score = self._calculate_escalation_score(conversation.messages)
        return escalation_score >= self.escalation_threshold

    def _calculate_escalation_score(self, messages: List[Message]) -> float:
        """
        Calculate the escalation score based on the conversation history.
        Uses weighted average with higher weights for recent messages.
        """
        if not messages:
            return 0.0
            
        # calculate scores w/ more weight on recent messages
        total_weight = 0
        weighted_score = 0
        
        for i, message in enumerate(messages):
            weight = 1 + (i / len(messages))
            total_weight += weight
            weighted_score += weight * self._get_message_escalation_score(message)
            
        return weighted_score / total_weight

    def _get_message_escalation_score(self, message: Message) -> float:
        """
        Calculate the escalation score for a single message using pattern matching.
        Returns scores based on content severity:
        1.0 - Self harm/suicide
        0.8 - Hate speech
        0.6 - Explicit content
        0.0 - Safe content
        """
        text = message.text.lower()
        
        if self._contains_self_harm_or_suicide(text):
            return 1.0
            
        if self._contains_hate_speech(text):
            return 0.8
            
        if self._contains_explicit_content(text):
            return 0.6
            
        return 0.0

    def _contains_explicit_content(self, text: str) -> bool:
        """
        Check if the given text contains explicit content using a multi-layered approach:
        1. Pattern matching for known explicit terms
        2. Context analysis for potentially innocent words used in explicit context
        3. Density checking to avoid false positives
        """
        text = text.lower()
        
        # count matches for explicit patterns
        pattern_matches = 0
        for pattern in self.explicit_content_patterns:
            matches = len(re.findall(pattern, text))
            pattern_matches += matches
        
        # calculate density of explicit content
        words = text.split()
        if not words:
            return False
            
        density = pattern_matches / len(words)
        
        # mark as explicit if density is high enough
        if density > 0.1:  # TODO: detemine if 10% threshold is enough/not
            return True
            
        # check combinations
        suspicious_combinations = [
            (r'\b(eat|lick)\b.*\b(out)\b', 0.7),
            (r'\b(come|came)\b.*\b(inside|in)\b', 0.8),
        ]
        
        for pattern, threshold in suspicious_combinations:
            if re.search(pattern, text) and density > threshold:
                return True
                
        return False

    def _contains_hate_speech(self, text: str) -> bool:
        """
        Detect hate speech using a weighted scoring system based on:
        1. Presence of hate speech indicators
        2. Contextual analysis
        3. Target identification
        
        This is based on research from papers like:
        "Automated Hate Speech Detection and the Problem of Offensive Language"
        """
        text = text.lower()
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        # remove stop words and calculate word frequencies
        words = [w for w in words if w not in stop_words]
        word_freq = Counter(words)
        
        # calculate hate speech score
        score = 0.0
        for word, weight in self.hate_speech_indicators:
            if word in word_freq:
                # Increase score based on frequency and weight
                score += word_freq[word] * weight
        
        # check patterns
        targeted_patterns = [
            (r'\b(all|those|these)\b.*\b(people|guys|folks)\b', 0.5),
            (r'\b(go back|leave)\b.*\b(country|place)\b', 0.7),
            (r'\b(hate|despise)\b.*\b(them|they|those)\b', 0.8)
        ]
        
        for pattern, weight in targeted_patterns:
            if re.search(pattern, text):
                score += weight
        
        # TODO: determine threshold
        return score >= 1.0

    def _contains_self_harm_or_suicide(self, text: str) -> bool:
        """
        Detect self-harm or suicidal ideation using techniques from crisis hotlines:
        1. Direct phrase matching
        2. Keyword density analysis
        3. Pattern recognition for subtle indicators
        
        Based on research from suicide prevention organizations and crisis hotlines
        """
        text = text.lower()
        
        # check for direct phrases first (highest priority)
        for phrase in self.self_harm_phrases:
            if phrase in text:
                return True
        
        # tokenize and analyze keywords
        words = word_tokenize(text)
        keywords_found = sum(1 for word in words if word in self.self_harm_keywords)
        
        # check keyword density
        if len(words) > 0:
            keyword_density = keywords_found / len(words)
            if keyword_density > 0.10:  # TODO: determine threshold
                return True
        
        # check for temporal indicators combined with negative emotions
        temporal_patterns = [
            (r"(today|tonight|tomorrow|soon).*?(die|end|over|gone)", 0.8),
            (r"(never|no\s+more|last).*?(wake|see|talk|speak)", 0.7),
            (r"(goodbye|bye|farewell).*?(everyone|world|all)", 0.9)
        ]
        
        for pattern, threshold in temporal_patterns:
            if re.search(pattern, text):
                return True
        
        # Look for planning indicators
        planning_patterns = [
            r"(wrote|writing|leave).*?(note|letter)",
            r"(give|giving).*?(away|stuff|things)",
            r"(made|making|have).*?(arrangements|plans)",
        ]
        
        planning_matches = sum(1 for pattern in planning_patterns if re.search(pattern, text))
        if planning_matches >= 2:
            return True
            
        return False

    def _format_prompt(self, messages: List[Message], user_message: str) -> str:
        """
        Format the conversation history and the user's message into a prompt for the therapy agent.
        """
        return f"Previous conversation:\n{self._format_conversation_history(messages)}\n\nClient: {user_message}\n\nRespond as the therapist:"

    def _format_conversation_history(self, messages: List[Message]) -> str:
        """
        Format the conversation history into a string.
        """
        return "\n".join([f"{msg.sender}: {msg.text}" for msg in messages[-5:]])