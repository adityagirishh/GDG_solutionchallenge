import numpy as np
import networkx as nx
import heapq
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from enhanced_bias_detector import EnhancedBiasDetector, DatasetSpecificEvaluator, StateOfArtMitigationTechniques

@dataclass
class EnhancedBiasScores:
    """Enhanced bias scores including all 7 bias types"""
    toxicity: float = 0.0
    sentiment: float = 0.0
    stereotypes: float = 0.0
    imbalance: float = 0.0
    context_shift: float = 0.0
    
    # 7 prominent bias types
    gender: float = 0.0
    racial_ethnic: float = 0.0
    religious: float = 0.0
    age: float = 0.0
    socioeconomic: float = 0.0
    disability: float = 0.0
    nationality_geographic: float = 0.0
    
    # Advanced metrics
    overall_bias: float = 0.0
    stereotype_intensity: float = 0.0
    contextual_bias: float = 0.0
    intersectional_bias: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return vars(self)

@dataclass
class EnhancedGraphNode:
    node_id: str
    text_content: str
    bias_scores: EnhancedBiasScores
    cbs: float  # Composite Bias Score
    crs: float  # Content Retention Score
    transformation_history: List[str]
    masked_segments: List[Tuple[int, int]]
    bias_explanation: Dict[str, Any] = None
    benchmark_scores: Dict[str, float] = None
    
    def __hash__(self):
        return hash(self.node_id)

class BenchmarkEvaluator:
    """Evaluate performance against bias detection benchmarks"""
    
    def __init__(self, detector: EnhancedBiasDetector):
        self.detector = detector
        self.dataset_evaluator = DatasetSpecificEvaluator(detector)
        self.benchmark_results = {}
    
    def evaluate_stereoset_performance(self, stereoset_data: List[Dict]) -> Dict[str, float]:
        """Evaluate performance on StereoSet-style data"""
        total_items = len(stereoset_data)
        correct_bias_detection = 0
        false_positives = 0
        false_negatives = 0
        
        all_bias_scores = []
        all_stereotype_scores = []
        all_anti_stereotype_scores = []
        
        for item in stereoset_data:
            results = self.dataset_evaluator.evaluate_stereoset_format(item)
            
            # Track scores
            all_bias_scores.append(results['bias_score'])
            all_stereotype_scores.append(results['stereotype_score'])
            all_anti_stereotype_scores.append(results['anti_stereotype_score'])
            
            # Determine if bias detection was correct
            if 'sentences' in item:
                for sentence_data in item['sentences']:
                    gold_label = sentence_data.get('gold_label', 'unrelated')
                    bias_scores = self.detector.compute_advanced_bias_scores(sentence_data.get('sentence', ''))
                    predicted_biased = bias_scores['overall_bias'] > 0.3
                    
                    if gold_label == 'stereotype' and predicted_biased:
                        correct_bias_detection += 1
                    elif gold_label == 'stereotype' and not predicted_biased:
                        false_negatives += 1
                    elif gold_label != 'stereotype' and predicted_biased:
                        false_positives += 1
        
        # Calculate metrics
        precision = correct_bias_detection / (correct_bias_detection + false_positives) if (correct_bias_detection + false_positives) > 0 else 0
        recall = correct_bias_detection / (correct_bias_detection + false_negatives) if (correct_bias_detection + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_bias_score': np.mean(all_bias_scores),
            'avg_stereotype_score': np.mean(all_stereotype_scores),
            'avg_anti_stereotype_score': np.mean(all_anti_stereotype_scores),
            'bias_score_std': np.std(all_bias_scores)
        }
    
    def evaluate_crows_pairs_performance(self, crows_pairs_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate performance on CrowS-Pairs data"""
        correct_rankings = 0
        total_pairs = len(crows_pairs_data)
        
        bias_differences = []
        ranking_confidence = []
        
        for more_biased, less_biased in crows_pairs_data:
            results = self.dataset_evaluator.evaluate_crows_pairs_format((more_biased, less_biased))
            
            if results['correct_ranking']:
                correct_rankings += 1
            
            bias_differences.append(abs(results['bias_difference']))
            ranking_confidence.append(results['bias_difference'])
        
        accuracy = correct_rankings / total_pairs if total_pairs > 0 else 0
        
        return {
            'ranking_accuracy': accuracy,
            'avg_bias_difference': np.mean(bias_differences),
            'bias_difference_std': np.std(bias_differences),
            'avg_ranking_confidence': np.mean(ranking_confidence),
            'correct_rankings': correct_rankings,
            'total_pairs': total_pairs
        }
    
    def evaluate_winobias_performance(self, winobias_data: List[Dict]) -> Dict[str, float]:
        """Evaluate performance on WinoBias data"""
        gender_bias_detected = 0
        total_items = len(winobias_data)
        
        gender_bias_scores = []
        pronoun_bias_scores = []
        
        for item in winobias_data:
            sentence = item.get('sentence', '')
            pronoun_resolution = item.get('pronoun_resolution', '')
            
            results = self.dataset_evaluator.evaluate_winobias_format(sentence, pronoun_resolution)
            
            gender_bias_scores.append(results['gender_bias'])
            pronoun_bias_scores.append(results['pronoun_bias'])
            
            if results['gender_bias'] > 0.3:
                gender_bias_detected += 1
        
        return {
            'gender_bias_detection_rate': gender_bias_detected / total_items if total_items > 0 else 0,
            'avg_gender_bias_score': np.mean(gender_bias_scores),
            'avg_pronoun_bias_score': np.mean(pronoun_bias_scores),
            'gender_bias_std': np.std(gender_bias_scores),
            'items_processed': total_items
        }

class EnhancedGraphOfThought:
    """Enhanced Graph of Thought system with benchmark evaluation"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.graph = nx.DiGraph()
        self.detector = EnhancedBiasDetector()
        self.benchmark_evaluator = BenchmarkEvaluator(self.detector)
        self.sota_techniques = StateOfArtMitigationTechniques()
        
        # Enhanced weights for all bias types
        self.weights = weights or {
            # Traditional metrics
            'toxicity': 0.05,
            'sentiment': 0.05,
            'stereotypes': 0.1,
            'imbalance': 0.05,
            'context_shift': 0.05,
            
            # 7 prominent bias types
            'gender': 0.15,
            'racial_ethnic': 0.15,
            'religious': 0.1,
            'age': 0.08,
            'socioeconomic': 0.08,
            'disability': 0.08,
            'nationality_geographic': 0.06,
            
            # Advanced metrics
            'overall_bias': 0.1,
            'stereotype_intensity': 0.05,
            'contextual_bias': 0.05,
            'intersectional_bias': 0.1
        }
        
        self.counter = 0
        self.traversal_history = []
        self.content_cache = set()
        self.similarity_threshold = 0.85
        
        # Benchmark comparison results
        self.comparison_results = {}
    
    def compute_enhanced_cbs(self, bias_scores: EnhancedBiasScores) -> float:
        """Compute Enhanced Composite Bias Score"""
        scores_dict = bias_scores.to_dict()
        weighted_sum = sum(self.weights.get(k, 0) * scores_dict.get(k, 0) for k in self.weights)
        return min(weighted_sum, 1.0)
    
    def create_enhanced_root_node(self, text: str) -> EnhancedGraphNode:
        """Create enhanced root node with comprehensive bias analysis"""
        node_id = f"node_{self.counter}"
        self.counter += 1
        
        # Compute comprehensive bias scores
        advanced_scores = self.detector.compute_advanced_bias_scores(text)
        
        # Create enhanced bias scores object
        bias_scores = EnhancedBiasScores(**advanced_scores)
        
        # Compute enhanced CBS
        cbs = self.compute_enhanced_cbs(bias_scores)
        
        # Generate bias explanation
        bias_explanation = self.detector.generate_bias_explanation(text, advanced_scores)
        
        node = EnhancedGraphNode(
            node_id=node_id,
            text_content=text,
            bias_scores=bias_scores,
            cbs=cbs,
            crs=1.0,
            transformation_history=[],
            masked_segments=[],
            bias_explanation=bias_explanation
        )
        
        self.graph.add_node(node_id, data=node)
        self.content_cache.add(text)
        
        return node
    
    def generate_enhanced_child_nodes(self, parent: EnhancedGraphNode, max_children: int = 8) -> List[EnhancedGraphNode]:
        """Generate child nodes with enhanced bias-specific filtering"""
        children = []
        
        # Enhanced filtering strategies targeting specific bias types
        strategies = [
            ('gender_bias_filtering', [0.3, 0.5, 0.7]),
            ('racial_ethnic_filtering', [0.3, 0.5, 0.7]),
            ('religious_bias_filtering', [0.4, 0.6]),
            ('age_bias_filtering', [0.4, 0.6]),
            ('socioeconomic_filtering', [0.4, 0.6]),
            ('disability_bias_filtering', [0.4, 0.6]),
            ('toxic_content_filtering', [0.5, 0.7, 0.9]),
            ('stereotype_intensity_reduction', [0.4, 0.6, 0.8])
        ]
        
        for filter_type, aggressiveness_levels in strategies:
            for aggressiveness in aggressiveness_levels:
                if len(children) >= max_children:
                    break
                
                # Apply bias-specific filtering
                filtered_text = self._apply_bias_specific_filtering(
                    parent.text_content, filter_type, aggressiveness
                )
                
                # Skip if no change or duplicate
                if (filtered_text == parent.text_content or 
                    self._is_duplicate_content(filtered_text)):
                    continue
                
                # Create child node
                child = self._create_child_node(parent, filtered_text, filter_type, aggressiveness)
                children.append(child)
        
        return children
    
    def _apply_bias_specific_filtering(self, text: str, filter_type: str, aggressiveness: float) -> str:
        """Apply bias-specific filtering strategies"""
        
        if filter_type == 'gender_bias_filtering':
            return self._filter_gender_bias(text, aggressiveness)
        elif filter_type == 'racial_ethnic_filtering':
            return self._filter_racial_ethnic_bias(text, aggressiveness)
        elif filter_type == 'religious_bias_filtering':
            return self._filter_religious_bias(text, aggressiveness)
        elif filter_type == 'age_bias_filtering':
            return self._filter_age_bias(text, aggressiveness)
        elif filter_type == 'socioeconomic_filtering':
            return self._filter_socioeconomic_bias(text, aggressiveness)
        elif filter_type == 'disability_bias_filtering':
            return self._filter_disability_bias(text, aggressiveness)
        elif filter_type == 'toxic_content_filtering':
            return self._filter_toxic_content(text, aggressiveness)
        elif filter_type == 'stereotype_intensity_reduction':
            return self._reduce_stereotype_intensity(text, aggressiveness)
        else:
            return text
    
    def _filter_gender_bias(self, text: str, aggressiveness: float) -> str:
        """Filter gender-specific bias"""
        gender_biased_patterns = [
            (r'\b(women|woman)\s+are\s+(naturally|inherently|typically)\s+(\w+)', 
             r'some \1 may be \3'),
            (r'\b(men|man)\s+are\s+(naturally|inherently|typically)\s+(\w+)', 
             r'some \1 may be \3'),
            (r'\ball\s+(women|men)\s+are\s+(\w+)', r'some \1 are \2'),
            (r'\b(he|she)\s+is\s+(obviously|clearly)\s+(\w+)', r'\1 is \3')
        ]
        
        filtered_text = text
        for pattern, replacement in gender_biased_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_racial_ethnic_bias(self, text: str, aggressiveness: float) -> str:
        """Filter racial/ethnic bias"""
        racial_biased_patterns = [
            (r'\b(black|white|asian|hispanic)\s+people\s+are\s+(all|always|typically)\s+(\w+)', 
             r'some \1 people may be \3'),
            (r'\ball\s+(blacks|whites|asians|hispanics)\s+are\s+(\w+)', 
             r'some \1 are \2'),
            (r'\b(\w+)\s+because\s+they\'re\s+(black|white|asian|hispanic)', 
             r'\1 for various reasons')
        ]
        
        filtered_text = text
        for pattern, replacement in racial_biased_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_religious_bias(self, text: str, aggressiveness: float) -> str:
        """Filter religious bias"""
        religious_biased_patterns = [
            (r'\b(muslims|christians|jews|hindus|buddhists)\s+are\s+(all|always)\s+(\w+)', 
             r'some \1 may be \3'),
            (r'\b(islam|christianity|judaism)\s+is\s+(obviously|clearly)\s+(\w+)', 
             r'\1 is considered by some to be \3')
        ]
        
        filtered_text = text
        for pattern, replacement in religious_biased_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_age_bias(self, text: str, aggressiveness: float) -> str:
        """Filter age-related bias"""
        age_biased_patterns = [
            (r'\b(young|old)\s+people\s+are\s+(all|always|typically)\s+(\w+)', 
             r'some \1 people may be \3'),
            (r'\b(millennials|boomers)\s+are\s+(all|always)\s+(\w+)', 
             r'some \1 are \3')
        ]
        
        filtered_text = text
        for pattern, replacement in age_biased_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_socioeconomic_bias(self, text: str, aggressiveness: float) -> str:
        """Filter socioeconomic bias"""
        socioeconomic_patterns = [
            (r'\b(poor|rich)\s+people\s+are\s+(all|always|typically)\s+(\w+)', 
             r'some \1 people may be \3'),
            (r'\bthe\s+(poor|wealthy)\s+are\s+(naturally|inherently)\s+(\w+)', 
             r'some \1 individuals may be \3')
        ]
        
        filtered_text = text
        for pattern, replacement in socioeconomic_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_disability_bias(self, text: str, aggressiveness: float) -> str:
        """Filter disability-related bias"""
        disability_patterns = [
            (r'\b(disabled|handicapped)\s+people\s+are\s+(all|always)\s+(\w+)', 
             r'some people with disabilities may be \3'),
            (r'\bpeople\s+with\s+(\w+)\s+are\s+(naturally|inherently)\s+(\w+)', 
             r'some people with \1 may be \3')
        ]
        
        filtered_text = text
        for pattern, replacement in disability_patterns:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _filter_toxic_content(self, text: str, aggressiveness: float) -> str:
        """Filter toxic content"""
        toxic_words = ['stupid', 'idiot', 'moron', 'disgusting', 'pathetic', 'worthless', 'terrible', 'awful']
        
        filtered_text = text
        for word in toxic_words:
            if np.random.rand() < aggressiveness:
                filtered_text = filtered_text.replace(word, '[FILTERED]')
        
        return filtered_text
    
    def _reduce_stereotype_intensity(self, text: str, aggressiveness: float) -> str:
        """Reduce stereotype intensity"""
        intensity_reducers = [
            (r'\ball\s+(\w+)\s+are\s+(\w+)', r'some \1 are \2'),
            (r'\bevery\s+(\w+)\s+is\s+(\w+)', r'some \1 are \2'),
            (r'\balways\s+(\w+)', r'sometimes \1'),
            (r'\bnever\s+(\w+)', r'rarely \1'),
            (r'\bobviously\s+(\w+)', r'\1'),
            (r'\bclearly\s+(\w+)', r'\1')
        ]
        
        filtered_text = text
        for pattern, replacement in intensity_reducers:
            if np.random.rand() < aggressiveness:
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def _is_duplicate_content(self, new_content: str) -> bool:
        """Check if content is too similar to existing content"""
        return new_content in self.content_cache
    
    def _create_child_node(self, parent: EnhancedGraphNode, filtered_text: str, 
                          filter_type: str, aggressiveness: float) -> EnhancedGraphNode:
        """Create a child node with comprehensive analysis"""
        node_id = f"node_{self.counter}"
        self.counter += 1
        
        # Compute comprehensive bias scores
        advanced_scores = self.detector.compute_advanced_bias_scores(filtered_text)
        bias_scores = EnhancedBiasScores(**advanced_scores)
        
        # Compute metrics
        cbs = self.compute_enhanced_cbs(bias_scores)
        crs = self._compute_crs(parent.text_content, filtered_text)
        
        # Generate bias explanation
        bias_explanation = self.detector.generate_bias_explanation(filtered_text, advanced_scores)
        
        transformation_step = f"{filter_type}_{aggressiveness}"
        
        child = EnhancedGraphNode(
            node_id=node_id,
            text_content=filtered_text,
            bias_scores=bias_scores,
            cbs=cbs,
            crs=crs,
            transformation_history=parent.transformation_history + [transformation_step],
            masked_segments=parent.masked_segments,
            bias_explanation=bias_explanation
        )
        
        self.graph.add_node(node_id, data=child)
        self.graph.add_edge(parent.node_id, node_id)
        self.content_cache.add(filtered_text)
        
        return child
    
    def _compute_crs(self, original: str, filtered: str) -> float:
        """Compute Content Retention Score"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, original, filtered).ratio()
    
    def compare_with_sota_techniques(self, text: str, bias_type: str = 'gender') -> Dict[str, Any]:
        """Compare Graph of Thought approach with state-of-the-art techniques"""
        
        # Run Graph of Thought
        root = self.create_enhanced_root_node(text)
        got_result = self.traverse_enhanced_graph(root, max_depth=4, bias_threshold=0.15)
        
        # Run state-of-the-art techniques
        sota_results = {}
        
        for technique_name, technique_func in self.sota_techniques.techniques.items():
            try:
                if technique_name == 'counterfactual_data_augmentation':
                    augmented_texts = technique_func(text, bias_type)
                    # Use the first augmented text for comparison
                    sota_text = augmented_texts[0] if augmented_texts else text
                else:
                    sota_text = technique_func(text)
                
                # Evaluate SOTA technique
                sota_scores = self.detector.compute_advanced_bias_scores(sota_text)
                sota_cbs = self.compute_enhanced_cbs(EnhancedBiasScores(**sota_scores))
                sota_crs = self._compute_crs(text, sota_text)
                
                sota_results[technique_name] = {
                    'text': sota_text,
                    'cbs': sota_cbs,
                    'crs': sota_crs,
                    'bias_scores': sota_scores
                }
                
            except Exception as e:
                print(f"Error running {technique_name}: {e}")
                sota_results[technique_name] = {
                    'text': text,
                    'cbs': root.cbs,
                    'crs': 1.0,
                    'error': str(e)
                }
        
        # Compare results
        comparison = {
            'original': {
                'text': text,
                'cbs': root.cbs,
                'crs': 1.0,
                'bias_scores': root.bias_scores.to_dict()
            },
            'graph_of_thought': {
                'text': got_result.text_content,
                'cbs': got_result.cbs,
                'crs': got_result.crs,
                'bias_scores': got_result.bias_scores.to_dict(),
                'transformation_history': got_result.transformation_history
            },
            'sota_techniques': sota_results
        }
        
        # Calculate relative improvements
        comparison['improvements'] = {}
        baseline_cbs = root.cbs
        
        # Graph of Thought improvement
        got_improvement = (baseline_cbs - got_result.cbs) / baseline_cbs * 100 if baseline_cbs > 0 else 0
        comparison['improvements']['graph_of_thought'] = {
            'bias_reduction_percent': got_improvement,
            'content_retention': got_result.crs
        }
        
        # SOTA techniques improvements
        for technique_name, results in sota_results.items():
            if 'error' not in results:
                sota_improvement = (baseline_cbs - results['cbs']) / baseline_cbs * 100 if baseline_cbs > 0 else 0
                comparison['improvements'][technique_name] = {
                    'bias_reduction_percent': sota_improvement,
                    'content_retention': results['crs']
                }
        
        self.comparison_results = comparison
        return comparison
    
    def traverse_enhanced_graph(self, root: EnhancedGraphNode, max_depth: int = 6, 
                               bias_threshold: float = 0.1) -> EnhancedGraphNode:
        """Enhanced graph traversal with multi-objective optimization"""
        # Priority queue: (negative_reward, tie_breaker, node, parent)
        pq = [(-self._compute_enhanced_reward(root), 0, root, None)]
        visited = set()
        best_node = root
        best_reward = self._compute_enhanced_reward(root)
        tie_breaker = 1
        
        for depth in range(max_depth):
            if not pq:
                break
            
            neg_reward, _, current_node, parent = heapq.heappop(pq)
            current_reward = -neg_reward
            
            if current_node.node_id in visited:
                continue
            
            visited.add(current_node.node_id)
            
            # Enhanced logging
            self.traversal_history.append({
                'depth': depth,
                'node_id': current_node.node_id,
                'cbs': current_node.cbs,
                'crs': current_node.crs,
                'reward': current_reward,
                'transformations': current_node.transformation_history,
                'primary_bias_types': [bt['type'] for bt in current_node.bias_explanation.get('primary_bias_types', [])],
                'bias_mechanisms': current_node.bias_explanation.get('bias_mechanisms', [])
            })
            
            # Update best node
            if current_reward > best_reward:
                best_node = current_node
                best_reward = current_reward
            
            # Early termination conditions
            if current_node.cbs < bias_threshold:
                logging.info(f"Early termination: CBS {current_node.cbs} below threshold {bias_threshold}")
                break
            
            # Generate children
            children = self.generate_enhanced_child_nodes(current_node)
            
            for child in children:
                if child.node_id not in visited:
                    child_reward = self._compute_enhanced_reward(child, current_node)
                    heapq.heappush(pq, (-child_reward, tie_breaker, child, current_node))
                    tie_breaker += 1
        
        return best_node
    
    def _compute_enhanced_reward(self, node: EnhancedGraphNode, parent: EnhancedGraphNode = None) -> float:
        """Compute enhanced reward considering multiple objectives"""
        # Base reward (lower bias = higher reward)
        base_reward = -node.cbs
        
        # Content retention bonus
        retention_bonus = node.crs * 0.5
        
        # Diversity bonus (reward diverse transformation paths)
        diversity_bonus = len(set(node.transformation_history)) * 0.1
        
        # Intersectionality penalty (multiple bias types are worse)
        intersectional_penalty = -node.bias_scores.intersectional_bias * 0.3
        
        if parent is not None:
            # Improvement reward
            improvement = parent.cbs - node.cbs
            improvement_reward = improvement * 2.0
            
            # Efficiency reward (prefer fewer transformations for same improvement)
            efficiency_bonus = improvement / max(len(node.transformation_history), 1) * 0.2
            
            return base_reward + retention_bonus + diversity_bonus + intersectional_penalty + improvement_reward + efficiency_bonus
        
        return base_reward + retention_bonus + diversity_bonus + intersectional_penalty
    
    def generate_comprehensive_report(self, node: EnhancedGraphNode, include_comparison: bool = True) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'node_analysis': {
                'node_id': node.node_id,
                'composite_bias_score': round(node.cbs, 4),
                'content_retention_score': round(node.crs, 4),
                'transformation_count': len(node.transformation_history),
                'transformation_history': node.transformation_history
            },
            
            'bias_breakdown': {
                'traditional_metrics': {
                    'toxicity': round(node.bias_scores.toxicity, 4),
                    'sentiment': round(node.bias_scores.sentiment, 4),
                    'stereotypes': round(node.bias_scores.stereotypes, 4),
                    'imbalance': round(node.bias_scores.imbalance, 4)
                },
                'seven_bias_types': {
                    'gender': round(node.bias_scores.gender, 4),
                    'racial_ethnic': round(node.bias_scores.racial_ethnic, 4),
                    'religious': round(node.bias_scores.religious, 4),
                    'age': round(node.bias_scores.age, 4),
                    'socioeconomic': round(node.bias_scores.socioeconomic, 4),
                    'disability': round(node.bias_scores.disability, 4),
                    'nationality_geographic': round(node.bias_scores.nationality_geographic, 4)
                },
                'advanced_metrics': {
                    'overall_bias': round(node.bias_scores.overall_bias, 4),
                    'stereotype_intensity': round(node.bias_scores.stereotype_intensity, 4),
                    'contextual_bias': round(node.bias_scores.contextual_bias, 4),
                    'intersectional_bias': round(node.bias_scores.intersectional_bias, 4)
                }
            },
            
            'bias_explanation': node.bias_explanation,
            
            'graph_statistics': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'exploration_depth': len(self.traversal_history),
                'content_variations_cached': len(self.content_cache)
            },
            
            'text_analysis': {
                'original_length': len(self.traversal_history[0]['text'] if self.traversal_history else ''),
                'final_length': len(node.text_content),
                'text_preview': node.text_content[:300] + '...' if len(node.text_content) > 300 else node.text_content
            }
        }
        
        if include_comparison and hasattr(self, 'comparison_results'):
            report['sota_comparison'] = self.comparison_results
        
        return report

# Import necessary libraries for the enhanced system
import re