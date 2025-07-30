import numpy as np
import re
from typing import Dict, List, Tuple, Set, Any
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import json

class EnhancedBiasDetector:
    """Enhanced bias detector targeting the 7 most prominent bias types"""
    
    def __init__(self):
        # Initialize embedding model for semantic analysis
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.embedding_model = None
            
        # Load comprehensive bias lexicons
        self.bias_lexicons = self._load_bias_lexicons()
        
        # Bias-specific patterns for advanced detection
        self.bias_patterns = self._load_bias_patterns()
        
        # Stereotype templates (based on StereoSet patterns)
        self.stereotype_templates = self._load_stereotype_templates()
        
        # Contextual bias indicators
        self.context_indicators = self._load_context_indicators()
    
    def _load_bias_lexicons(self) -> Dict[str, Set[str]]:
        """Load comprehensive bias lexicons for each bias type"""
        return {
            'gender': {
                # Explicit gender terms
                'male', 'female', 'man', 'woman', 'boy', 'girl', 'men', 'women',
                'masculine', 'feminine', 'he', 'she', 'his', 'her', 'him',
                # Gendered professions (stereotypical)
                'nurse', 'teacher', 'secretary', 'housewife', 'engineer', 'doctor',
                'ceo', 'programmer', 'mechanic', 'construction worker',
                # Gendered traits
                'emotional', 'aggressive', 'nurturing', 'competitive', 'sensitive'
            },
            
            'racial_ethnic': {
                # Racial/ethnic identifiers
                'black', 'white', 'asian', 'hispanic', 'latino', 'african', 'european',
                'arab', 'indian', 'native', 'indigenous', 'caucasian', 'oriental',
                # Nationality-based
                'american', 'chinese', 'mexican', 'japanese', 'german', 'italian',
                # Stereotypical traits
                'articulate', 'exotic', 'foreign', 'urban', 'thuggish', 'model minority'
            },
            
            'religious': {
                'christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist',
                'catholic', 'protestant', 'islamic', 'judaism', 'christianity',
                'fundamentalist', 'extremist', 'devout', 'secular', 'religious'
            },
            
            'age': {
                'young', 'old', 'elderly', 'senior', 'teenager', 'millennial', 'boomer',
                'middle-aged', 'youth', 'geriatric', 'adolescent', 'juvenile',
                'mature', 'aged', 'ancient', 'experienced', 'inexperienced'
            },
            
            'socioeconomic': {
                'poor', 'rich', 'wealthy', 'homeless', 'privileged', 'underprivileged',
                'upper-class', 'middle-class', 'lower-class', 'elite', 'working-class',
                'educated', 'uneducated', 'sophisticated', 'ghetto', 'trailer trash'
            },
            
            'disability': {
                'disabled', 'handicapped', 'blind', 'deaf', 'mentally ill', 'retarded',
                'autistic', 'wheelchair', 'impaired', 'special needs', 'abnormal',
                'defective', 'broken', 'invalid', 'crippled', 'normal'
            },
            
            'nationality_geographic': {
                'urban', 'rural', 'suburban', 'inner-city', 'country', 'city',
                'southern', 'northern', 'coastal', 'midwest', 'foreign', 'immigrant',
                'native', 'local', 'outsider', 'cosmopolitan', 'provincial'
            }
        }
    
    def _load_bias_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for detecting biased language structures"""
        return {
            'stereotype_attribution': [
                r'\b(\w+)\s+(?:are|is)\s+(?:all|always|typically|usually|naturally|inherently)\s+(\w+)',
                r'\b(\w+)\s+(?:tend to be|are often|are generally)\s+(\w+)',
                r'(?:all|most|many)\s+(\w+)\s+(?:are|were)\s+(\w+)',
            ],
            
            'comparative_bias': [
                r'\b(\w+)\s+(?:are|is)\s+(?:better|worse|superior|inferior)\s+(?:than|to)\s+(\w+)',
                r'\b(\w+)\s+(?:more|less)\s+(\w+)\s+than\s+(\w+)',
            ],
            
            'essentialist': [
                r'\b(\w+)\s+(?:by nature|naturally|inherently|genetically)\s+(\w+)',
                r'(?:it\'s|its)\s+(?:natural|normal|typical)\s+for\s+(\w+)\s+to\s+(\w+)',
            ],
            
            'exclusionary': [
                r'\bonly\s+(\w+)\s+(?:can|should|are able to)\s+(\w+)',
                r'\b(\w+)\s+(?:don\'t|cannot|shouldn\'t)\s+(\w+)',
            ]
        }
    
    def _load_stereotype_templates(self) -> Dict[str, List[str]]:
        """Load stereotype templates based on StereoSet format"""
        return {
            'gender': [
                "Women are {trait}",
                "Men are {trait}",
                "{profession} is a job for {gender}",
                "{gender} are better at {skill}"
            ],
            
            'racial_ethnic': [
                "{race} people are {trait}",
                "{race} are {stereotype}",
                "All {nationality} are {trait}"
            ],
            
            'religious': [
                "{religion} are {trait}",
                "People who practice {religion} are {stereotype}"
            ],
            
            'age': [
                "{age_group} are {trait}",
                "{age_group} people {behavior}"
            ],
            
            'socioeconomic': [
                "{class} people are {trait}",
                "The {class} are {stereotype}"
            ],
            
            'disability': [
                "People with {disability} are {trait}",
                "{disability} people {behavior}"
            ]
        }
    
    def _load_context_indicators(self) -> Dict[str, List[str]]:
        """Load contextual indicators that suggest bias"""
        return {
            'qualification_words': ['but', 'however', 'surprisingly', 'actually', 'even'],
            'intensifiers': ['very', 'extremely', 'completely', 'totally', 'absolutely'],
            'generalizers': ['all', 'every', 'always', 'never', 'none', 'everyone'],
            'hedges': ['some', 'many', 'often', 'usually', 'typically', 'generally']
        }
    
    def compute_advanced_bias_scores(self, text: str) -> Dict[str, float]:
        """Compute advanced bias scores for all 7 bias types"""
        scores = {}
        
        for bias_type in self.bias_lexicons.keys():
            scores[bias_type] = self._compute_bias_type_score(text, bias_type)
        
        # Aggregate scores
        scores['overall_bias'] = np.mean(list(scores.values()))
        scores['max_bias'] = max(scores.values())
        scores['stereotype_intensity'] = self._compute_stereotype_intensity(text)
        scores['contextual_bias'] = self._compute_contextual_bias(text)
        
        return scores
    
    def _compute_bias_type_score(self, text: str, bias_type: str) -> float:
        """Compute bias score for a specific bias type"""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        # Lexicon-based scoring
        lexicon_score = self._compute_lexicon_score(words, bias_type)
        
        # Pattern-based scoring
        pattern_score = self._compute_pattern_score(text, bias_type)
        
        # Semantic similarity scoring (if embedding model available)
        semantic_score = self._compute_semantic_score(text, bias_type)
        
        # Weighted combination
        weights = {
            'lexicon': 0.4,
            'pattern': 0.4,
            'semantic': 0.2
        }
        
        final_score = (
            weights['lexicon'] * lexicon_score +
            weights['pattern'] * pattern_score +
            weights['semantic'] * semantic_score
        )
        
        return min(final_score, 1.0)
    
    def _compute_lexicon_score(self, words: List[str], bias_type: str) -> float:
        """Compute score based on bias lexicon matches"""
        lexicon = self.bias_lexicons[bias_type]
        matches = sum(1 for word in words if word in lexicon)
        return matches / len(words) if words else 0.0
    
    def _compute_pattern_score(self, text: str, bias_type: str) -> float:
        """Compute score based on biased language patterns"""
        pattern_matches = 0
        total_patterns = 0
        
        for pattern_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                total_patterns += 1
                if re.search(pattern, text, re.IGNORECASE):
                    # Check if the match involves terms from this bias type
                    if self._pattern_involves_bias_type(pattern, text, bias_type):
                        pattern_matches += 1
        
        return pattern_matches / max(total_patterns, 1)
    
    def _pattern_involves_bias_type(self, pattern: str, text: str, bias_type: str) -> bool:
        """Check if a pattern match involves terms from the specified bias type"""
        matches = re.finditer(pattern, text, re.IGNORECASE)
        lexicon = self.bias_lexicons[bias_type]
        
        for match in matches:
            match_text = match.group().lower()
            if any(term in match_text for term in lexicon):
                return True
        return False
    
    def _compute_semantic_score(self, text: str, bias_type: str) -> float:
        """Compute semantic similarity to bias stereotypes"""
        if not self.embedding_model:
            return 0.0
        
        try:
            text_embedding = self.embedding_model.encode(text)
            
            # Generate stereotype sentences for this bias type
            stereotype_sentences = self._generate_stereotype_sentences(bias_type)
            
            max_similarity = 0.0
            for sentence in stereotype_sentences:
                sentence_embedding = self.embedding_model.encode(sentence)
                similarity = 1 - cosine(text_embedding, sentence_embedding)
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
        except:
            return 0.0
    
    def _generate_stereotype_sentences(self, bias_type: str) -> List[str]:
        """Generate stereotype sentences for semantic comparison"""
        sentences = []
        templates = self.stereotype_templates.get(bias_type, [])
        lexicon = self.bias_lexicons[bias_type]
        
        # Sample terms for template filling
        sample_terms = list(lexicon)[:10]  # Use first 10 terms
        
        for template in templates:
            for term in sample_terms:
                try:
                    # Simple template filling - in practice, use more sophisticated methods
                    filled = template.format(
                        trait=term, 
                        profession=term, 
                        gender=term,
                        race=term,
                        nationality=term,
                        religion=term,
                        age_group=term,
                        class=term,
                        disability=term,
                        stereotype=term,
                        skill=term,
                        behavior=term
                    )
                    sentences.append(filled)
                except KeyError:
                    continue
        
        return sentences[:20]  # Return top 20 for efficiency
    
    def _compute_stereotype_intensity(self, text: str) -> float:
        """Compute overall stereotype intensity"""
        intensity_indicators = [
            'all', 'every', 'always', 'never', 'completely', 'totally',
            'absolutely', 'definitely', 'obviously', 'clearly'
        ]
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        intensity_count = sum(1 for word in words if word in intensity_indicators)
        return min(intensity_count / len(words) * 10, 1.0)  # Scale up for visibility
    
    def _compute_contextual_bias(self, text: str) -> float:
        """Compute contextual bias based on linguistic patterns"""
        context_score = 0.0
        
        # Check for qualifying language that might indicate bias
        qualifiers = self.context_indicators['qualification_words']
        qualifier_count = sum(1 for word in qualifiers if word in text.lower())
        
        # Check for generalizing language
        generalizers = self.context_indicators['generalizers']
        generalizer_count = sum(1 for word in generalizers if word in text.lower())
        
        # Normalize by text length
        words = text.split()
        if words:
            context_score = (qualifier_count + generalizer_count * 2) / len(words)
        
        return min(context_score * 5, 1.0)  # Scale for visibility
    
    def detect_bias_intersectionality(self, text: str) -> Dict[str, List[str]]:
        """Detect intersectional bias (multiple bias types in same context)"""
        intersections = {}
        text_lower = text.lower()
        
        # Find co-occurring bias types
        bias_presence = {}
        for bias_type, lexicon in self.bias_lexicons.items():
            bias_presence[bias_type] = [word for word in lexicon if word in text_lower]
        
        # Identify intersections
        active_biases = [bt for bt, words in bias_presence.items() if words]
        
        if len(active_biases) > 1:
            for i, bias1 in enumerate(active_biases):
                for bias2 in active_biases[i+1:]:
                    intersection_key = f"{bias1}_{bias2}"
                    intersections[intersection_key] = {
                        'bias_types': [bias1, bias2],
                        'terms_found': bias_presence[bias1] + bias_presence[bias2]
                    }
        
        return intersections
    
    def generate_bias_explanation(self, text: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed explanation of detected bias"""
        explanation = {
            'overall_assessment': self._get_bias_level_description(scores['overall_bias']),
            'primary_bias_types': [],
            'problematic_phrases': [],
            'bias_mechanisms': [],
            'intersectional_issues': self.detect_bias_intersectionality(text)
        }
        
        # Identify primary bias types
        bias_threshold = 0.3
        for bias_type in self.bias_lexicons.keys():
            if scores.get(bias_type, 0) > bias_threshold:
                explanation['primary_bias_types'].append({
                    'type': bias_type,
                    'score': scores[bias_type],
                    'severity': self._get_bias_level_description(scores[bias_type])
                })
        
        # Extract problematic phrases
        explanation['problematic_phrases'] = self._extract_problematic_phrases(text)
        
        # Identify bias mechanisms
        explanation['bias_mechanisms'] = self._identify_bias_mechanisms(text)
        
        return explanation
    
    def _get_bias_level_description(self, score: float) -> str:
        """Get descriptive text for bias level"""
        if score < 0.1:
            return "Minimal bias detected"
        elif score < 0.3:
            return "Low bias detected"
        elif score < 0.5:
            return "Moderate bias detected"
        elif score < 0.7:
            return "High bias detected"
        else:
            return "Severe bias detected"
    
    def _extract_problematic_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract specific phrases that contribute to bias"""
        problematic = []
        
        # Check stereotype patterns
        for pattern_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    problematic.append({
                        'phrase': match.group(),
                        'type': pattern_type,
                        'position': match.span(),
                        'confidence': 0.8  # Pattern-based matches are high confidence
                    })
        
        return problematic
    
    def _identify_bias_mechanisms(self, text: str) -> List[str]:
        """Identify the mechanisms through which bias is expressed"""
        mechanisms = []
        text_lower = text.lower()
        
        # Stereotyping
        if any(word in text_lower for word in ['all', 'every', 'always', 'never']):
            mechanisms.append("Overgeneralization/Stereotyping")
        
        # Essentializing
        if any(phrase in text_lower for phrase in ['by nature', 'naturally', 'inherently']):
            mechanisms.append("Essentializing")
        
        # Comparative bias
        if any(word in text_lower for word in ['better', 'worse', 'superior', 'inferior']):
            mechanisms.append("Hierarchical comparison")
        
        # Othering
        if any(word in text_lower for word in ['them', 'they', 'those people', 'different']):
            mechanisms.append("Othering/In-group vs Out-group")
        
        return mechanisms


class DatasetSpecificEvaluator:
    """Evaluator specifically designed for bias benchmark datasets"""
    
    def __init__(self, detector: EnhancedBiasDetector):
        self.detector = detector
    
    def evaluate_stereoset_format(self, data: Dict) -> Dict[str, float]:
        """Evaluate bias for StereoSet format data"""
        results = {
            'bias_score': 0.0,
            'stereotype_score': 0.0,
            'anti_stereotype_score': 0.0,
            'unrelated_score': 0.0
        }
        
        if 'sentences' in data:
            for sentence_data in data['sentences']:
                sentence = sentence_data.get('sentence', '')
                gold_label = sentence_data.get('gold_label', 'unrelated')
                
                bias_scores = self.detector.compute_advanced_bias_scores(sentence)
                overall_bias = bias_scores['overall_bias']
                
                if gold_label == 'stereotype':
                    results['stereotype_score'] += overall_bias
                elif gold_label == 'anti-stereotype':
                    results['anti_stereotype_score'] += overall_bias
                else:
                    results['unrelated_score'] += overall_bias
        
        # Normalize by count
        sentence_count = len(data.get('sentences', []))
        if sentence_count > 0:
            for key in results:
                results[key] /= sentence_count
        
        return results
    
    def evaluate_crows_pairs_format(self, sentence_pair: Tuple[str, str]) -> Dict[str, Any]:
        """Evaluate bias for CrowS-Pairs format data"""
        sent_more_biased, sent_less_biased = sentence_pair
        
        bias_more = self.detector.compute_advanced_bias_scores(sent_more_biased)
        bias_less = self.detector.compute_advanced_bias_scores(sent_less_biased)
        
        return {
            'more_biased_score': bias_more['overall_bias'],
            'less_biased_score': bias_less['overall_bias'],
            'correct_ranking': bias_more['overall_bias'] > bias_less['overall_bias'],
            'bias_difference': bias_more['overall_bias'] - bias_less['overall_bias'],
            'detailed_more': bias_more,
            'detailed_less': bias_less
        }
    
    def evaluate_winobias_format(self, sentence: str, pronoun_resolution: str) -> Dict[str, Any]:
        """Evaluate bias for WinoBias format data"""
        bias_scores = self.detector.compute_advanced_bias_scores(sentence)
        
        # Specific gender bias analysis
        gender_bias = bias_scores.get('gender', 0.0)
        
        # Analyze pronoun resolution for stereotypical associations
        pronoun_bias = self._analyze_pronoun_bias(sentence, pronoun_resolution)
        
        return {
            'overall_bias': bias_scores['overall_bias'],
            'gender_bias': gender_bias,
            'pronoun_bias': pronoun_bias,
            'stereotype_intensity': bias_scores['stereotype_intensity'],
            'bias_explanation': self.detector.generate_bias_explanation(sentence, bias_scores)
        }
    
    def _analyze_pronoun_bias(self, sentence: str, resolution: str) -> float:
        """Analyze bias in pronoun resolution"""
        # Check if resolution aligns with stereotypical professions
        stereotypical_male = ['engineer', 'doctor', 'ceo', 'manager', 'programmer']
        stereotypical_female = ['nurse', 'teacher', 'secretary', 'assistant']
        
        resolution_lower = resolution.lower()
        sentence_lower = sentence.lower()
        
        bias_score = 0.0
        
        # Check for stereotypical associations
        if any(prof in sentence_lower for prof in stereotypical_male):
            if 'he' in resolution_lower or 'his' in resolution_lower:
                bias_score += 0.5
        
        if any(prof in sentence_lower for prof in stereotypical_female):
            if 'she' in resolution_lower or 'her' in resolution_lower:
                bias_score += 0.5
        
        return min(bias_score, 1.0)


# State-of-the-art mitigation techniques for comparison
class StateOfArtMitigationTechniques:
    """Implementation of current best practices for bias mitigation"""
    
    def __init__(self):
        self.techniques = {
            'counterfactual_data_augmentation': self.counterfactual_augmentation,
            'adversarial_debiasing': self.adversarial_debiasing,
            'fairness_constraints': self.fairness_constraints,
            'bias_aware_fine_tuning': self.bias_aware_fine_tuning,
            'demographic_parity_postprocessing': self.demographic_parity_postprocessing,
            'equalized_odds_adjustment': self.equalized_odds_adjustment,
            'calibration_debiasing': self.calibration_debiasing
        }
    
    def counterfactual_augmentation(self, text: str, bias_type: str) -> List[str]:
        """Generate counterfactual examples by swapping bias-related terms"""
        counterfactuals = []
        
        if bias_type == 'gender':
            swaps = [('he', 'she'), ('his', 'her'), ('him', 'her'), ('man', 'woman'), ('men', 'women')]
        elif bias_type == 'racial_ethnic':
            swaps = [('black', 'white'), ('african american', 'european american'), ('asian', 'white')]
        else:
            return [text]  # No swaps defined for this bias type
        
        for original, replacement in swaps:
            cf_text = text.replace(original, replacement)
            if cf_text != text:
                counterfactuals.append(cf_text)
        
        return counterfactuals if counterfactuals else [text]
    
    def adversarial_debiasing(self, text: str) -> str:
        """Simulate adversarial debiasing (placeholder for complex implementation)"""
        # This would involve training adversarial networks
        # For now, return a simplified version
        return text.replace('[BIAS_DETECTED]', '[CONTENT_MODIFIED]')
    
    def fairness_constraints(self, text: str, constraint_type: str = 'demographic_parity') -> str:
        """Apply fairness constraints during processing"""
        # Placeholder for constraint-based optimization
        return text
    
    def bias_aware_fine_tuning(self, text: str) -> str:
        """Simulate bias-aware fine-tuning approach"""
        # This would involve model fine-tuning with bias-aware loss
        return text
    
    def demographic_parity_postprocessing(self, text: str) -> str:
        """Apply demographic parity post-processing"""
        # Ensure equal representation across groups
        return text
    
    def equalized_odds_adjustment(self, text: str) -> str:
        """Apply equalized odds adjustment"""
        # Ensure equal true positive and false positive rates
        return text
    
    def calibration_debiasing(self, text: str) -> str:
        """Apply calibration-based debiasing"""
        # Ensure predictions are well-calibrated across groups
        return text