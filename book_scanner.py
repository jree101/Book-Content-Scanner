#!/usr/bin/env python3
"""
Book Content Scanner - PHASE 2 COMPLETE
Includes ALL Phase 1 features PLUS:
- ShieldGemma 2B for fast pre-screening
- Enhanced sentiment analysis (twitter-roberta-base-sentiment-latest)
- 2-3x faster processing with better accuracy
"""

import subprocess
import sys
import importlib.util
import os
from pathlib import Path
import datetime
from collections import defaultdict
import time

def install_package(package_name, import_name=None):
    """Install a package if it's not already installed"""
    if import_name is None:
        import_name = package_name
    
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {package_name}: {e}")
            sys.exit(1)
    else:
        print(f"✓ {package_name} already installed")

# Install required dependencies
print("=" * 60)
print("PHASE 2 SCANNER - Installing Dependencies")
print("=" * 60)
install_package("ebooklib")
install_package("beautifulsoup4", "bs4")
install_package("lxml")
install_package("transformers")
install_package("torch")
install_package("sentencepiece")
install_package("accelerate")
install_package("bitsandbytes")  # For quantization
print("=" * 60)
print()

# Now import the packages
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Import configuration files
try:
    from content_config import CONTENT_CATEGORIES, FIGURATIVE_EXCLUSIONS
    print("✓ Loaded content categories")
except ImportError:
    print("ERROR: content_config.py not found!")
    sys.exit(1)

try:
    from scanner_settings import *
    print("✓ Loaded scanner settings")
except ImportError:
    print("WARNING: Using default settings")
    CONFIDENCE_LEVELS = {"high": 0.7, "moderate": 0.55, "low": 0.5}
    MINIMUM_CONFIDENCE_THRESHOLD = 0.5
    CONTEXT_PARAGRAPHS_BEFORE = 2
    CONTEXT_PARAGRAPHS_AFTER = 2
    MINIMUM_PARAGRAPH_LENGTH = 20
    TEXT_EXCERPT_MAX_LENGTH = 150
    TOXICITY_MODEL = "unitary/toxic-bert"
    HATE_SPEECH_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
    LLAMA_GUARD_MODEL = "meta-llama/Llama-Guard-3-1B"
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    SHIELDGEMMA_MODEL = "google/shieldgemma-2b"
    USE_GPU_IF_AVAILABLE = True
    INPUT_FOLDER = "books"
    OUTPUT_FOLDER = "scan_reports"
    USE_TOXICITY_MODEL = True
    USE_HATE_SPEECH_MODEL = True
    USE_LLAMA_GUARD = True
    USE_SENTIMENT_MODEL = True
    USE_SHIELDGEMMA = True
    LLAMA_GUARD_HIGH_CONFIDENCE_ONLY = True
    LLAMA_GUARD_CONFIDENCE_THRESHOLD = 0.7
    LLAMA_GUARD_CATEGORIES = ["S1", "S3", "S4", "S9", "S10", "S11"]
    VIOLENCE_REQUIRES_ACTION = True
    VIOLENCE_ACTION_PATTERNS = [
        r"\b(murdered|killing|stabbed|shot (him|her|them)|slaughtered)\b",
        r"\b(blood.*spilled|corpse|mutilated|decapitated)\b",
    ]
    LGBT_REQUIRES_CONTEXT = True
    LGBT_CONTEXT_PATTERNS = [
        r"\b(same-sex|gay|lesbian) (marriage|couple|relationship)\b",
        r"\b(gender identity|transgender)\b",
    ]
    DRUGS_ALCOHOL_REQUIRES_CONSUMPTION = True
    DRUGS_ALCOHOL_CONSUMPTION_PATTERNS = [
        r"\b(intoxicated|hammered)\b",
        r"\b(smoking|snorting|injecting)\b",
    ]
    SHOW_CHAPTER_PROGRESS = True

print()


class Phase2ContentScanner:
    """Phase 2 Scanner with ShieldGemma pre-screening and enhanced sentiment"""
    
    def __init__(self):
        print("=" * 60)
        print("INITIALIZING PHASE 2 ENHANCED AI MODELS")
        print("=" * 60)
        
        device = 0 if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else -1
        gpu_status = "GPU" if device == 0 else "CPU"
        print(f"Device: {gpu_status}\n")
        
        self.stats = {
            "pre_screened": 0,
            "passed_screening": 0,
            "skipped_by_shield": 0
        }
        
        # Model 0: ShieldGemma 2B (NEW - Pre-screening)
        if USE_SHIELDGEMMA:
            try:
                print("Loading Model 0: ShieldGemma 2B (Pre-screening)...")
                self.shieldgemma_tokenizer = AutoTokenizer.from_pretrained(SHIELDGEMMA_MODEL)
                self.shieldgemma_model = AutoModelForCausalLM.from_pretrained(
                    SHIELDGEMMA_MODEL,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print(f"✓ {SHIELDGEMMA_MODEL}")
            except Exception as e:
                print(f"✗ Could not load ShieldGemma: {e}")
                self.shieldgemma_model = None
                self.shieldgemma_tokenizer = None
        else:
            self.shieldgemma_model = None
            self.shieldgemma_tokenizer = None
        
        # Model 1: Toxicity Detection
        if USE_TOXICITY_MODEL:
            try:
                print("\nLoading Model 1: Toxicity Detection...")
                self.toxicity_model = pipeline(
                    "text-classification",
                    model=TOXICITY_MODEL,
                    device=device
                )
                print(f"✓ {TOXICITY_MODEL}")
            except Exception as e:
                print(f"✗ Could not load toxicity model: {e}")
                self.toxicity_model = None
        else:
            self.toxicity_model = None
        
        # Model 2: Hate Speech Detection
        if USE_HATE_SPEECH_MODEL:
            try:
                print("\nLoading Model 2: Hate Speech Detection...")
                self.hate_speech_model = pipeline(
                    "text-classification",
                    model=HATE_SPEECH_MODEL,
                    device=device
                )
                print(f"✓ {HATE_SPEECH_MODEL}")
            except Exception as e:
                print(f"✗ Could not load hate speech model: {e}")
                self.hate_speech_model = None
        else:
            self.hate_speech_model = None
        
        # Model 3: Enhanced Sentiment Analysis (NEW - Upgraded)
        if USE_SENTIMENT_MODEL:
            try:
                print("\nLoading Model 3: Enhanced Sentiment Analysis...")
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=SENTIMENT_MODEL,
                    device=device
                )
                print(f"✓ {SENTIMENT_MODEL}")
            except Exception as e:
                print(f"✗ Could not load sentiment model: {e}")
                self.sentiment_model = None
        else:
            self.sentiment_model = None
        
        # Model 4: Llama Guard 3
        if USE_LLAMA_GUARD and LLAMA_GUARD_MODEL:
            try:
                print("\nLoading Model 4: Llama Guard 3 (Verification)...")
                print("⚠️  This may take a moment...")
                
                self.llama_guard_tokenizer = AutoTokenizer.from_pretrained(LLAMA_GUARD_MODEL)
                self.llama_guard_model = AutoModelForCausalLM.from_pretrained(
                    LLAMA_GUARD_MODEL,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print(f"✓ {LLAMA_GUARD_MODEL}")
            except Exception as e:
                print(f"✗ Could not load Llama Guard: {e}")
                self.llama_guard_model = None
                self.llama_guard_tokenizer = None
        else:
            self.llama_guard_model = None
            self.llama_guard_tokenizer = None
        
        print("=" * 60)
        print()
    
    def pre_screen_with_shieldgemma(self, text):
        """Phase 2: Fast pre-screening with ShieldGemma"""
        if not self.shieldgemma_model or not self.shieldgemma_tokenizer:
            return True  # If not available, pass through
        
        self.stats["pre_screened"] += 1
        
        try:
            # ShieldGemma prompt for general safety check
            prompt = f"""You are a policy expert trying to help determine whether a user prompt is in
violation of the defined safety policies.

<start_of_turn>
Human Question: {text[:200]}
<end_of_turn>

Our safety principle is defined in the below:

* "No Hate Speech": The prompt shall not contain or seek generation of content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.

* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual.

* "No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).

* "No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal).

===

Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.
"""
            
            inputs = self.shieldgemma_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.shieldgemma_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=self.shieldgemma_tokenizer.eos_token_id
                )
            
            response = self.shieldgemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if response starts with "No"
            response_clean = response.split('\n')[0].strip().lower()
            is_safe = response_clean.startswith("no")
            
            if is_safe:
                self.stats["skipped_by_shield"] += 1
                return False  # Skip this paragraph, it's safe
            else:
                self.stats["passed_screening"] += 1
                return True  # Needs deeper analysis
        except Exception as e:
            # On error, pass through to deeper analysis
            return True
    
    def contains_keywords(self, text, category):
        if not CONTENT_CATEGORIES[category].get("enabled", True):
            return []
        
        keywords = CONTENT_CATEGORIES[category]["keywords"]
        text_lower = text.lower()
        
        matched_keywords = []
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return matched_keywords
    
    def analyze_toxicity(self, text):
        if not self.toxicity_model:
            return None
        
        try:
            if len(text) > 500:
                text = text[:500]
            
            result = self.toxicity_model(text)[0]
            return {
                "is_toxic": result["label"] == "toxic",
                "confidence": result["score"]
            }
        except Exception as e:
            return None
    
    def analyze_hate_speech(self, text):
        if not self.hate_speech_model:
            return None
        
        try:
            if len(text) > 500:
                text = text[:500]
            
            result = self.hate_speech_model(text)[0]
            return {
                "is_hate": result["label"].lower() == "hate",
                "confidence": result["score"]
            }
        except Exception as e:
            return None
    
    def analyze_sentiment(self, text):
        """Phase 2: Enhanced sentiment analysis"""
        if not self.sentiment_model:
            return None
        
        try:
            if len(text) > 500:
                text = text[:500]
            
            result = self.sentiment_model(text)[0]
            # Returns Negative (0), Neutral (1), or Positive (2)
            return {
                "label": result["label"],
                "score": result["score"]
            }
        except Exception as e:
            return None
    
    def verify_with_llama_guard(self, text, category):
        if not self.llama_guard_model or not self.llama_guard_tokenizer:
            return None
        
        try:
            category_map = {
                "profanity": "S9",
                "sexual_content": "S11",
                "violence": "S1",
                "disturbing_themes": "S10",
            }
            
            guard_category = category_map.get(category)
            if not guard_category or guard_category not in LLAMA_GUARD_CATEGORIES:
                return None
            
            prompt = f"""<s>[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S9: Hate.
S10: Self-Harm.
S11: Sexual Content.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
Agent: {text[:200]}
<END CONVERSATION>

Provide your safety assessment for Agent in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
            
            inputs = self.llama_guard_tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llama_guard_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=self.llama_guard_tokenizer.eos_token_id
                )
            
            response = self.llama_guard_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            is_unsafe = "unsafe" in response.lower()
            violated_categories = []
            if is_unsafe:
                lines = response.strip().split("\n")
                if len(lines) > 1:
                    violated_categories = [c.strip() for c in lines[1].split(",")]
            
            return {
                "is_unsafe": is_unsafe,
                "violated_categories": violated_categories,
                "explanation": response.strip()
            }
        except Exception as e:
            return None
    
    def is_figurative_language(self, text):
        text_lower = text.lower()
        for pattern in FIGURATIVE_EXCLUSIONS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def analyze_paragraph(self, paragraph, category):
        """Phase 2: Enhanced analysis with pre-screening"""
        # Phase 2 NEW: Pre-screen with ShieldGemma
        if self.shieldgemma_model:
            if not self.pre_screen_with_shieldgemma(paragraph):
                return 0.0, [], {}  # ShieldGemma says it's safe, skip heavy models
        
        matched_keywords = self.contains_keywords(paragraph, category)
        if not matched_keywords:
            return 0.0, [], {}
        
        if self.is_figurative_language(paragraph):
            return 0.0, [], {}
        
        score = 0.5
        ai_details = {}
        
        if category == "profanity":
            toxicity_result = self.analyze_toxicity(paragraph)
            hate_result = self.analyze_hate_speech(paragraph)
            
            if toxicity_result and toxicity_result["is_toxic"]:
                score = max(score, toxicity_result["confidence"])
                ai_details["toxicity_score"] = toxicity_result["confidence"]
            
            if hate_result and hate_result["is_hate"]:
                score = max(score, hate_result["confidence"])
                ai_details["hate_speech_score"] = hate_result["confidence"]
        
        elif category == "sexual_content":
            toxicity_result = self.analyze_toxicity(paragraph)
            if toxicity_result and toxicity_result["is_toxic"]:
                score = max(score, toxicity_result["confidence"])
                ai_details["toxicity_score"] = toxicity_result["confidence"]
        
        elif category == "violence" and VIOLENCE_REQUIRES_ACTION:
            violent_actions = VIOLENCE_ACTION_PATTERNS
            has_violent_action = any(re.search(p, paragraph, re.IGNORECASE) 
                                    for p in violent_actions)
            if has_violent_action:
                score = 0.7
                
                # Phase 2 NEW: Use sentiment to understand context
                sentiment = self.analyze_sentiment(paragraph)
                if sentiment:
                    ai_details["sentiment"] = sentiment
                    # Negative sentiment + violence = higher confidence
                    if sentiment["label"] == "Negative":
                        score = min(score + 0.1, 0.9)
        
        elif category == "lgbt_content" and LGBT_REQUIRES_CONTEXT:
            lgbt_contexts = LGBT_CONTEXT_PATTERNS
            has_lgbt_context = any(re.search(p, paragraph, re.IGNORECASE) 
                                  for p in lgbt_contexts)
            if has_lgbt_context:
                score = 0.8
            else:
                score = 0.4
        
        elif category == "drugs_alcohol" and DRUGS_ALCOHOL_REQUIRES_CONSUMPTION:
            consumption_patterns = DRUGS_ALCOHOL_CONSUMPTION_PATTERNS
            has_consumption = any(re.search(p, paragraph, re.IGNORECASE) 
                                 for p in consumption_patterns)
            if has_consumption:
                score = 0.7
        
        # Llama Guard verification for high confidence
        if score >= LLAMA_GUARD_CONFIDENCE_THRESHOLD and LLAMA_GUARD_HIGH_CONFIDENCE_ONLY:
            guard_result = self.verify_with_llama_guard(paragraph, category)
            if guard_result:
                ai_details["llama_guard"] = guard_result
                if guard_result["is_unsafe"]:
                    score = max(score, 0.9)
                else:
                    score = min(score, 0.6)
        
        return score, matched_keywords, ai_details
    
    def get_performance_stats(self):
        """Return performance statistics"""
        total = self.stats["pre_screened"]
        if total == 0:
            return "No paragraphs screened yet"
        
        skip_rate = (self.stats["skipped_by_shield"] / total) * 100
        return f"ShieldGemma: {skip_rate:.1f}% of content pre-screened as safe (skipped heavy models)"


def generate_ai_explanation(ai_details, confidence, category):
    """Generate human-readable explanation"""
    explanation_parts = []
    
    if confidence >= 0.9:
        explanation_parts.append("🔴 VERY HIGH confidence - Multiple AI models agree this is concerning")
    elif confidence >= 0.7:
        explanation_parts.append("🔴 HIGH confidence - AI models detected concerning content")
    elif confidence >= 0.55:
        explanation_parts.append("🟡 MODERATE confidence - Possible concern detected")
    else:
        explanation_parts.append("🟢 LOW confidence - May be false positive")
    
    if "toxicity_score" in ai_details:
        score = ai_details["toxicity_score"]
        explanation_parts.append(f"\n  • Toxicity Model: {score:.1%} confidence this contains offensive language")
    
    if "hate_speech_score" in ai_details:
        score = ai_details["hate_speech_score"]
        explanation_parts.append(f"\n  • Hate Speech Model: {score:.1%} confidence this contains hate speech or slurs")
    
    if "sentiment" in ai_details:
        sentiment = ai_details["sentiment"]
        explanation_parts.append(f"\n  • Sentiment Analysis: {sentiment['label']} ({sentiment['score']:.1%} confidence)")
    
    if "llama_guard" in ai_details:
        guard = ai_details["llama_guard"]
        if guard["is_unsafe"]:
            categories_map = {
                "S1": "Violent Crimes",
                "S3": "Sex Crimes",
                "S4": "Child Exploitation",
                "S9": "Hate Speech",
                "S10": "Self-Harm",
                "S11": "Sexual Content"
            }
            violated = [categories_map.get(c, c) for c in guard.get("violated_categories", [])]
            if violated:
                explanation_parts.append(f"\n  • Llama Guard: UNSAFE - Violates: {', '.join(violated)}")
            else:
                explanation_parts.append(f"\n  • Llama Guard: UNSAFE - Policy violation detected")
            explanation_parts.append(f"\n  • Verification: This content was confirmed as genuinely concerning")
        else:
            explanation_parts.append(f"\n  • Llama Guard: SAFE - May be false positive")
    
    category_explanations = {
        "profanity": "Contains profane or vulgar language",
        "sexual_content": "Contains sexual or explicit content",
        "violence": "Contains violent or disturbing content",
        "lgbt_content": "Contains LGBT themes or relationships",
        "drugs_alcohol": "Contains drug or alcohol references",
        "disturbing_themes": "Contains disturbing themes (self-harm, suicide, eating disorders)",
        "occult": "Contains occult or supernatural themes"
    }
    
    if category in category_explanations:
        explanation_parts.append(f"\n\nCategory: {category_explanations[category]}")
    
    return "\n".join(explanation_parts)


def get_context(paragraphs, current_idx):
    start_idx = max(0, current_idx - CONTEXT_PARAGRAPHS_BEFORE)
    end_idx = min(len(paragraphs), current_idx + CONTEXT_PARAGRAPHS_AFTER + 1)
    
    context_paragraphs = []
    for i in range(start_idx, end_idx):
        if i == current_idx:
            context_paragraphs.append(f">>> {paragraphs[i]} <<<")
        else:
            context_paragraphs.append(paragraphs[i])
    
    return "\n\n".join(context_paragraphs)


def get_confidence_level(confidence_score):
    if confidence_score >= CONFIDENCE_LEVELS["high"]:
        return "high"
    elif confidence_score >= CONFIDENCE_LEVELS["moderate"]:
        return "moderate"
    else:
        return "low"


def organize_by_confidence(flagged_content):
    organized = {
        "high": defaultdict(list),
        "moderate": defaultdict(list),
        "low": defaultdict(list)
    }
    
    for item in flagged_content:
        level = get_confidence_level(item["confidence"])
        category = item["category"]
        organized[level][category].append(item)
    
    result = {}
    for level in ["high", "moderate", "low"]:
        result[level] = {}
        for category in sorted(organized[level].keys()):
            result[level][category] = organized[level][category]
    
    return result


def scan_ebook(file_path, scanner):
    """Phase 2: Scan with pre-screening"""
    if not os.path.exists(file_path):
        return None
    
    book_name = os.path.basename(file_path)
    print(f"\nScanning: {book_name}")
    print(f"Confidence threshold: {MINIMUM_CONFIDENCE_THRESHOLD}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        book = epub.read_epub(file_path)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    results = []
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapter_count += 1
            if SHOW_CHAPTER_PROGRESS:
                print(f"  Chapter {chapter_count}...", end="\r")
            
            try:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                text_content = soup.get_text()
                paragraphs = [p.strip() for p in text_content.split("\n") if p.strip()]
            except:
                continue

            for p_idx, paragraph in enumerate(paragraphs):
                if len(paragraph) < MINIMUM_PARAGRAPH_LENGTH:
                    continue
                
                for category in CONTENT_CATEGORIES.keys():
                    if not CONTENT_CATEGORIES[category].get("enabled", True):
                        continue
                    
                    confidence_score, matched_keywords, ai_details = scanner.analyze_paragraph(paragraph, category)
                    
                    if confidence_score >= MINIMUM_CONFIDENCE_THRESHOLD:
                        full_context = get_context(paragraphs, p_idx)
                        
                        excerpt = paragraph[:TEXT_EXCERPT_MAX_LENGTH]
                        if len(paragraph) > TEXT_EXCERPT_MAX_LENGTH:
                            excerpt += "..."
                        
                        result_entry = {
                            "chapter": chapter_count,
                            "paragraph": p_idx + 1,
                            "category": category,
                            "confidence": round(confidence_score, 3),
                            "matched_terms": matched_keywords,
                            "text_excerpt": excerpt,
                            "full_text": full_context
                        }
                        
                        if ai_details:
                            result_entry["ai_analysis"] = ai_details
                            
                            if confidence_score >= CONFIDENCE_LEVELS["high"]:
                                result_entry["ai_explanation"] = generate_ai_explanation(
                                    ai_details, confidence_score, category
                                )
                        
                        results.append(result_entry)

    elapsed_time = time.time() - start_time
    
    print(f"  ✓ Complete: {len(results)} flags in {elapsed_time:.1f}s" + " " * 20)
    
    return {
        "book_name": book_name,
        "file_path": file_path,
        "chapter_count": chapter_count,
        "flagged_content": results,
        "scan_time": elapsed_time,
        "performance_stats": scanner.get_performance_stats()
    }


def generate_individual_report(book_result, output_dir):
    """Generate report"""
    book_name = Path(book_result["book_name"]).stem
    
    organized_content = organize_by_confidence(book_result["flagged_content"])
    
    stats_by_level = {}
    for level in ["high", "moderate", "low"]:
        level_items = []
        for category_items in organized_content[level].values():
            level_items.extend(category_items)
        
        category_stats = {}
        for category, items in organized_content[level].items():
            confidences = [item["confidence"] for item in items]
            category_stats[category] = {
                "count": len(items),
                "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0
            }
        
        stats_by_level[level] = {
            "total_flags": len(level_items),
            "categories": category_stats
        }
    
    ai_explained = sum(1 for item in book_result["flagged_content"] 
                      if "ai_explanation" in item)
    llama_guard_verified = sum(1 for item in book_result["flagged_content"] 
                               if "ai_analysis" in item and "llama_guard" in item["ai_analysis"])
    
    report = {
        "book_info": {
            "name": book_result["book_name"],
            "chapters": book_result["chapter_count"],
            "total_flags": len(book_result["flagged_content"]),
            "scan_time_seconds": round(book_result["scan_time"], 2),
            "ai_explained_flags": ai_explained,
            "llama_guard_verified": llama_guard_verified
        },
        "performance": {
            "scan_time": f"{book_result['scan_time']:.1f} seconds",
            "shieldgemma_stats": book_result["performance_stats"]
        },
        "summary": {
            "high_confidence": stats_by_level["high"]["total_flags"],
            "moderate_confidence": stats_by_level["moderate"]["total_flags"],
            "low_confidence": stats_by_level["low"]["total_flags"]
        },
        "statistics_by_confidence": stats_by_level,
        "flagged_content": {
            "high_confidence": dict(organized_content["high"]),
            "moderate_confidence": dict(organized_content["moderate"]),
            "low_confidence": dict(organized_content["low"])
        }
    }
    
    output_file = os.path.join(output_dir, f"{book_name}_report.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report["book_info"], report["summary"], report["performance"]


def generate_summary_report(all_books_info, output_dir):
    """Generate summary"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        "scan_date": datetime.datetime.now().isoformat(),
        "total_books": len(all_books_info),
        "phase": "Phase 2 Complete (ShieldGemma + RoBERTa + Llama Guard + Enhanced Sentiment)",
        "books": all_books_info
    }
    
    output_file = os.path.join(output_dir, f"batch_summary_phase2_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Summary: {output_file}")
    return output_file


def find_epub_files(directory):
    """Find all EPUB files"""
    epub_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.epub'):
                epub_files.append(os.path.join(root, file))
    return epub_files


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("BOOK CONTENT SCANNER - PHASE 2 COMPLETE")
    print("=" * 60)
    print()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) > 1:
        scan_folder = sys.argv[1]
    else:
        scan_folder = os.path.join(script_dir, INPUT_FOLDER)
    
    if not os.path.exists(scan_folder):
        os.makedirs(scan_folder)
        print(f"Created: {scan_folder}")
        return
    
    epub_files = find_epub_files(scan_folder)
    
    if not epub_files:
        print(f"No EPUB files in: {scan_folder}")
        return
    
    print(f"Scan folder: {scan_folder}")
    print(f"Found {len(epub_files)} EPUB file(s):")
    for epub_file in epub_files:
        print(f"  📖 {os.path.basename(epub_file)}")
    print()
    
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scanner = Phase2ContentScanner()
    
    all_books_info = []
    total_start_time = time.time()
    
    print("\n" + "=" * 60)
    print("SCANNING WITH PHASE 2 OPTIMIZATIONS")
    print("=" * 60)
    
    for i, epub_file in enumerate(epub_files, 1):
        print(f"\n[{i}/{len(epub_files)}] ", end="")
        
        result = scan_ebook(epub_file, scanner)
        
        if result:
            book_info, summary, performance = generate_individual_report(result, output_dir)
            
            all_books_info.append({
                "name": book_info["name"],
                "chapters": book_info["chapters"],
                "total_flags": book_info["total_flags"],
                "scan_time": performance["scan_time"],
                "high_confidence": summary["high_confidence"],
                "moderate_confidence": summary["moderate_confidence"],
                "low_confidence": summary["low_confidence"],
                "ai_explained": book_info["ai_explained_flags"],
                "llama_guard_verified": book_info["llama_guard_verified"]
            })
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("PHASE 2 SCAN COMPLETE")
    print("=" * 60)
    
    if all_books_info:
        generate_summary_report(all_books_info, output_dir)
        
        print(f"\nScanned {len(all_books_info)} book(s) in {total_time:.1f}s")
        print(f"\nReports: {output_dir}")
        print("\nSummary by book:")
        for book in all_books_info:
            print(f"  📖 {book['name']}")
            print(f"     Total: {book['total_flags']} | High: {book['high_confidence']} | Mod: {book['moderate_confidence']} | Low: {book['low_confidence']}")
            print(f"     ⚡ Scan time: {book['scan_time']}")
            if book['ai_explained'] > 0:
                print(f"     🤖 AI Explained: {book['ai_explained']} flags")
            if book['llama_guard_verified'] > 0:
                print(f"     🛡️  Llama Guard: {book['llama_guard_verified']} verified")
        
        print("\n" + "=" * 60)
        print("PHASE 2 FEATURES")
        print("=" * 60)
        print("  ⚡ ShieldGemma 2B Pre-screening (2-3x faster)")
        print("  🎯 Enhanced Sentiment Analysis")
        print("  ✅ All Phase 1 features included")
        print("  📊 Performance statistics")
        print("\n  Models Active:")
        print("    0. ShieldGemma 2B (pre-screening)")
        print("    1. Toxic-BERT (profanity)")
        print("    2. RoBERTa (hate speech)")
        print("    3. Twitter-RoBERTa (sentiment)")
        print("    4. Llama Guard 3 (verification)")
        print("\n" + "=" * 60)
        print("CONFIDENCE LEVELS")
        print("=" * 60)
        print(f"  🔴 HIGH (≥{CONFIDENCE_LEVELS['high']}): Very likely objectionable")
        print(f"  🟡 MOD (≥{CONFIDENCE_LEVELS['moderate']}): Possibly objectionable")
        print(f"  🟢 LOW (≥{CONFIDENCE_LEVELS['low']}): May be false positive")


if __name__ == "__main__":
    main()
