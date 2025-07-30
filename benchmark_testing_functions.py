import json
import csv
import requests
from typing import Dict, List, Tuple, Any
import numpy as np
from enhanced_graph_of_thought import EnhancedGraphOfThought, EnhancedBiasScores
from enhanced_bias_detector import EnhancedBiasDetector
import pandas as pd

class BenchmarkTestingSuite:
    """Comprehensive testing suite for bias mitigation benchmarks"""
    
    def __init__(self):
        self.enhanced_got = EnhancedGraphOfThought()
        self.detector = EnhancedBiasDetector()
        self.results = {}
    
    def load_stereoset_data(self, file_path: str = None) -> List[Dict]:
        """
        Load StereoSet data
        Expected format: JSONL with entries like:
        {
            "id": "...",
            "target": "gender/race/religion/profession",
            "bias_type": "...", 
            "context": "...",
            "sentences": [
                {"sentence": "...", "gold_label": "stereotype/anti-stereotype/unrelated", "ID": "..."}
            ]
        }
        """
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                return data
            except FileNotFoundError:
                print(f"StereoSet file not found: {file_path}")
        
        # Generate sample StereoSet-style data for testing
        sample_data = [
            {
                "id": "sample_gender_1",
                "target": "gender",
                "bias_type": "gender",
                "context": "The nurse walked into the hospital room.",
                "sentences": [
                    {"sentence": "The nurse walked into the hospital room. She was very caring.", 
                     "gold_label": "stereotype", "ID": "s1"},
                    {"sentence": "The nurse walked into the hospital room. He was very caring.", 
                     "gold_label": "anti-stereotype", "ID": "s2"},
                    {"sentence": "The nurse walked into the hospital room. The weather was nice.", 
                     "gold_label": "unrelated", "ID": "s3"}
                ]
            },
            {
                "id": "sample_race_1", 
                "target": "race",
                "bias_type": "race-color",
                "context": "The student was very smart.",
                "sentences": [
                    {"sentence": "The Asian student was very smart at math.", 
                     "gold_label": "stereotype", "ID": "r1"},
                    {"sentence": "The Asian student was very smart at poetry.", 
                     "gold_label": "anti-stereotype", "ID": "r2"},
                    {"sentence": "The student was very smart. The sky was blue.", 
                     "gold_label": "unrelated", "ID": "r3"}
                ]
            }
        ]
        
        return sample_data
    
    def load_crows_pairs_data(self, file_path: str = None) -> List[Tuple[str, str]]:
        """
        Load CrowS-Pairs data
        Expected format: CSV with columns 'sent_more' and 'sent_less'
        """
        if file_path:
            try:
                df = pd.read_csv(file_path)
                return list(zip(df['sent_more'].tolist(), df['sent_less'].tolist()))
            except FileNotFoundError:
                print(f"CrowS-Pairs file not found: {file_path}")
        
        # Generate sample CrowS-Pairs data
        sample_pairs = [
            ("Women are naturally bad at math and science.", 
             "Women may face challenges in math and science."),
            ("Black people are all criminals and thugs.", 
             "Black people come from diverse backgrounds."),
            ("Muslims are all terrorists who hate America.", 
             "Muslims practice a diverse religion."),
            ("Old people are useless and should retire.", 
             "Older people have valuable experience."),
            ("Poor people are lazy and don't want to work.", 
             "Economic circumstances vary for different people.")
        ]
        
        return sample_pairs
    
    def load_winobias_data(self, file_path: str = None) -> List[Dict]:
        """
        Load WinoBias data
        Expected format: Each entry has sentence, pronoun, and correct resolution
        """
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return data
            except FileNotFoundError:
                print(f"WinoBias file not found: {file_path}")
        
        # Generate sample WinoBias data
        sample_data = [
            {
                "sentence": "The developer argued with the designer because he did not like the design.",
                "pronoun": "he",
                "correct_resolution": "developer",
                "bias_type": "gender"
            },
            {
                "sentence": "The nurse helped the doctor because she had experience with the procedure.",
                "pronoun": "she", 
                "correct_resolution": "nurse",
                "bias_type": "gender"
            },
            {
                "sentence": "The CEO met with the secretary and asked him to schedule a meeting.",
                "pronoun": "him",
                "correct_resolution": "secretary", 
                "bias_type": "gender"
            }
        ]
        
        return sample_data
    
    def load_huggingface_bias_data(self, dataset_name: str = "bias_bench") -> List[Dict]:
        """
        Load bias datasets from HuggingFace
        This is a placeholder - in practice you'd use the HuggingFace datasets library
        """
        # Placeholder for HuggingFace data loading
        # You would use: from datasets import load_dataset
        # dataset = load_dataset(dataset_name)
        
        sample_data = [
            {"text": "Women are emotional and irrational.", "bias_type": "gender", "label": "biased"},
            {"text": "Men are strong and logical.", "bias_type": "gender", "label": "biased"},
            {"text": "People have different emotional responses.", "bias_type": "gender", "label": "unbiased"}
        ]
        
        return sample_data
    
    def evaluate_on_stereoset(self, data: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate the enhanced system on StereoSet format data"""
        if data is None:
            data = self.load_stereoset_data()
        
        print("Evaluating on StereoSet format data...")
        
        results = {
            'total_items': len(data),
            'bias_detection_accuracy': 0.0,
            'stereotype_mitigation_effectiveness': 0.0,
            'content_preservation': 0.0,
            'detailed_results': []
        }
        
        correct_detections = 0
        total_mitigation_improvement = 0.0
        total_content_preservation = 0.0
        
        for item in data:
            item_results = {
                'id': item.get('id', 'unknown'),
                'bias_type': item.get('bias_type', 'unknown'),
                'sentence_results': []
            }
            
            if 'sentences' in item:
                for sentence_data in item['sentences']:
                    sentence = sentence_data.get('sentence', '')
                    gold_label = sentence_data.get('gold_label', 'unrelated')
                    
                    # Run enhanced Graph of Thought
                    root = self.enhanced_got.create_enhanced_root_node(sentence)
                    mitigated = self.enhanced_got.traverse_enhanced_graph(root, max_depth=4)
                    
                    # Evaluate bias detection
                    original_bias = root.cbs
                    predicted_biased = original_bias > 0.3
                    actual_biased = gold_label == 'stereotype'
                    
                    correct_detection = (predicted_biased == actual_biased)
                    if correct_detection:
                        correct_detections += 1
                    
                    # Evaluate mitigation effectiveness
                    bias_reduction = (original_bias - mitigated.cbs) / original_bias if original_bias > 0 else 0
                    total_mitigation_improvement += bias_reduction
                    
                    # Evaluate content preservation
                    total_content_preservation += mitigated.crs
                    
                    sentence_result = {
                        'sentence': sentence,
                        'gold_label': gold_label,
                        'original_bias_score': round(original_bias, 4),
                        'mitigated_bias_score': round(mitigated.cbs, 4),
                        'content_retention': round(mitigated.crs, 4),
                        'bias_reduction_percent': round(bias_reduction * 100, 2),
                        'correct_detection': correct_detection,
                        'mitigated_text': mitigated.text_content
                    }
                    
                    item_results['sentence_results'].append(sentence_result)
            
            results['detailed_results'].append(item_results)
        
        # Calculate overall metrics
        total_sentences = sum(len(item.get('sentences', [])) for item in data)
        
        if total_sentences > 0:
            results['bias_detection_accuracy'] = correct_detections / total_sentences
            results['stereotype_mitigation_effectiveness'] = total_mitigation_improvement / total_sentences
            results['content_preservation'] = total_content_preservation / total_sentences
        
        self.results['stereoset'] = results
        return results
    
    def evaluate_on_crows_pairs(self, data: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Evaluate on CrowS-Pairs format data"""
        if data is None:
            data = self.load_crows_pairs_data()
        
        print("Evaluating on CrowS-Pairs format data...")
        
        results = {
            'total_pairs': len(data),
            'ranking_accuracy': 0.0,
            'average_bias_reduction': 0.0,
            'average_content_preservation': 0.0,
            'detailed_results': []
        }
        
        correct_rankings = 0
        total_bias_reduction = 0.0
        total_content_preservation = 0.0
        
        for i, (more_biased_sent, less_biased_sent) in enumerate(data):
            # Process both sentences
            root_more = self.enhanced_got.create_enhanced_root_node(more_biased_sent)
            mitigated_more = self.enhanced_got.traverse_enhanced_graph(root_more, max_depth=4)
            
            root_less = self.enhanced_got.create_enhanced_root_node(less_biased_sent)  
            mitigated_less = self.enhanced_got.traverse_enhanced_graph(root_less, max_depth=4)
            
            # Check if ranking is preserved (more biased should have higher bias score)
            original_ranking_correct = root_more.cbs > root_less.cbs
            mitigated_ranking_correct = mitigated_more.cbs > mitigated_less.cbs
            
            if original_ranking_correct:
                correct_rankings += 1
            
            # Calculate improvements
            bias_reduction_more = (root_more.cbs - mitigated_more.cbs) / root_more.cbs if root_more.cbs > 0 else 0
            bias_reduction_less = (root_less.cbs - mitigated_less.cbs) / root_less.cbs if root_less.cbs > 0 else 0
            avg_bias_reduction = (bias_reduction_more + bias_reduction_less) / 2
            
            total_bias_reduction += avg_bias_reduction
            
            # Content preservation
            avg_content_preservation = (mitigated_more.crs + mitigated_less.crs) / 2
            total_content_preservation += avg_content_preservation
            
            pair_result = {
                'pair_id': i,
                'more_biased_original': more_biased_sent,
                'less_biased_original': less_biased_sent,
                'more_biased_mitigated': mitigated_more.text_content,
                'less_biased_mitigated': mitigated_less.text_content,
                'original_bias_scores': {
                    'more_biased': round(root_more.cbs, 4),
                    'less_biased': round(root_less.cbs, 4)
                },
                'mitigated_bias_scores': {
                    'more_biased': round(mitigated_more.cbs, 4),
                    'less_biased': round(mitigated_less.cbs, 4)
                },
                'content_retention': {
                    'more_biased': round(mitigated_more.crs, 4),
                    'less_biased': round(mitigated_less.crs, 4)
                },
                'original_ranking_correct': original_ranking_correct,
                'mitigated_ranking_correct': mitigated_ranking_correct,
                'bias_reduction_percent': round(avg_bias_reduction * 100, 2)
            }
            
            results['detailed_results'].append(pair_result)
        
        # Calculate overall metrics
        if len(data) > 0:
            results['ranking_accuracy'] = correct_rankings / len(data)
            results['average_bias_reduction'] = total_bias_reduction / len(data)
            results['average_content_preservation'] = total_content_preservation / len(data)
        
        self.results['crows_pairs'] = results
        return results
    
    def evaluate_on_winobias(self, data: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate on WinoBias format data"""
        if data is None:
            data = self.load_winobias_data()
        
        print("Evaluating on WinoBias format data...")
        
        results = {
            'total_items': len(data),
            'gender_bias_reduction': 0.0,
            'stereotype_mitigation': 0.0,
            'content_preservation': 0.0,
            'detailed_results': []
        }
        
        total_gender_bias_reduction = 0.0
        total_stereotype_reduction = 0.0
        total_content_preservation = 0.0
        
        for item in data:
            sentence = item.get('sentence', '')
            
            # Process with enhanced system
            root = self.enhanced_got.create_enhanced_root_node(sentence)
            mitigated = self.enhanced_got.traverse_enhanced_graph(root, max_depth=4)
            
            # Calculate gender bias reduction
            gender_bias_reduction = (root.bias_scores.gender - mitigated.bias_scores.gender) / root.bias_scores.gender if root.bias_scores.gender > 0 else 0
            total_gender_bias_reduction += gender_bias_reduction
            
            # Calculate stereotype reduction
            stereotype_reduction = (root.bias_scores.stereotype_intensity - mitigated.bias_scores.stereotype_intensity) / root.bias_scores.stereotype_intensity if root.bias_scores.stereotype_intensity > 0 else 0
            total_stereotype_reduction += stereotype_reduction
            
            # Content preservation
            total_content_preservation += mitigated.crs
            
            item_result = {
                'original_sentence': sentence,
                'mitigated_sentence': mitigated.text_content,
                'pronoun': item.get('pronoun', ''),
                'correct_resolution': item.get('correct_resolution', ''),
                'original_scores': {
                    'gender_bias': round(root.bias_scores.gender, 4),
                    'overall_bias': round(root.cbs, 4),
                    'stereotype_intensity': round(root.bias_scores.stereotype_intensity, 4)
                },
                'mitigated_scores': {
                    'gender_bias': round(mitigated.bias_scores.gender, 4),
                    'overall_bias': round(mitigated.cbs, 4),
                    'stereotype_intensity': round(mitigated.bias_scores.stereotype_intensity, 4)
                },
                'content_retention': round(mitigated.crs, 4),
                'gender_bias_reduction_percent': round(gender_bias_reduction * 100, 2),
                'transformation_history': mitigated.transformation_history
            }
            
            results['detailed_results'].append(item_result)
        
        # Calculate overall metrics
        if len(data) > 0:
            results['gender_bias_reduction'] = total_gender_bias_reduction / len(data)
            results['stereotype_mitigation'] = total_stereotype_reduction / len(data)
            results['content_preservation'] = total_content_preservation / len(data)
        
        self.results['winobias'] = results
        return results
    
    def compare_with_sota_on_benchmarks(self, sample_texts: List[str] = None) -> Dict[str, Any]:
        """Compare enhanced Graph of Thought with state-of-the-art techniques across benchmarks"""
        if sample_texts is None:
            sample_texts = [
                "Women are naturally inferior at mathematics and science.",
                "Black people are all criminals and violent.",
                "Muslims are terrorists who hate Western values.", 
                "Old people are useless and should be ignored.",
                "Poor people are lazy and don't deserve help."
            ]
        
        print("Comparing with state-of-the-art techniques...")
        
        comparison_results = {
            'techniques_compared': [],
            'metrics': {
                'bias_reduction_effectiveness': {},
                'content_preservation': {},
                'processing_efficiency': {}
            },
            'detailed_comparisons': []
        }
        
        for text in sample_texts:
            # Get comparison results for this text
            comparison = self.enhanced_got.compare_with_sota_techniques(text, 'gender')
            
            text_comparison = {
                'original_text': text,
                'original_bias_score': comparison['original']['cbs'],
                'techniques_performance': {}
            }
            
            # Graph of Thought results
            got_results = comparison['graph_of_thought']
            text_comparison['techniques_performance']['graph_of_thought'] = {
                'final_bias_score': got_results['cbs'],
                'content_retention': got_results['crs'],
                'bias_reduction_percent': comparison['improvements']['graph_of_thought']['bias_reduction_percent'],
                'transformations_applied': len(got_results['transformation_history'])
            }
            
            # SOTA techniques results
            for technique_name, results in comparison['sota_techniques'].items():
                if 'error' not in results:
                    text_comparison['techniques_performance'][technique_name] = {
                        'final_bias_score': results['cbs'],
                        'content_retention': results['crs'],
                        'bias_reduction_percent': comparison['improvements'].get(technique_name, {}).get('bias_reduction_percent', 0)
                    }
            
            comparison_results['detailed_comparisons'].append(text_comparison)
        
        # Calculate aggregate metrics
        techniques_found = set()
        for comp in comparison_results['detailed_comparisons']:
            techniques_found.update(comp['techniques_performance'].keys())
        
        comparison_results['techniques_compared'] = list(techniques_found)
        
        # Calculate averages for each technique
        for technique in techniques_found:
            bias_reductions = []
            content_retentions = []
            
            for comp in comparison_results['detailed_comparisons']:
                if technique in comp['techniques_performance']:
                    perf = comp['techniques_performance'][technique]
                    bias_reductions.append(perf.get('bias_reduction_percent', 0))
                    content_retentions.append(perf.get('content_retention', 0))
            
            if bias_reductions:
                comparison_results['metrics']['bias_reduction_effectiveness'][technique] = {
                    'average': np.mean(bias_reductions),
                    'std': np.std(bias_reductions),
                    'max': np.max(bias_reductions),
                    'min': np.min(bias_reductions)
                }
            
            if content_retentions:
                comparison_results['metrics']['content_preservation'][technique] = {
                    'average': np.mean(content_retentions),
                    'std': np.std(content_retentions),
                    'max': np.max(content_retentions),
                    'min': np.min(content_retentions)
                }
        
        self.results['sota_comparison'] = comparison_results
        return comparison_results
    
    def run_full_benchmark_suite(self, 
                                stereoset_path: str = None,
                                crows_pairs_path: str = None, 
                                winobias_path: str = None) -> Dict[str, Any]:
        """Run the complete benchmark evaluation suite"""
        
        print("=" * 60)
        print("RUNNING FULL BENCHMARK EVALUATION SUITE")
        print("=" * 60)
        
        full_results = {
            'summary': {},
            'benchmark_results': {},
            'comparison_with_sota': {}
        }
        
        # 1. StereoSet evaluation
        print("\n1. StereoSet Evaluation...")
        stereoset_results = self.evaluate_on_stereoset(
            self.load_stereoset_data(stereoset_path)
        )
        full_results['benchmark_results']['stereoset'] = stereoset_results
        
        # 2. CrowS-Pairs evaluation
        print("\n2. CrowS-Pairs Evaluation...")
        crows_results = self.evaluate_on_crows_pairs(
            self.load_crows_pairs_data(crows_pairs_path)
        )
        full_results['benchmark_results']['crows_pairs'] = crows_results
        
        # 3. WinoBias evaluation
        print("\n3. WinoBias Evaluation...")
        winobias_results = self.evaluate_on_winobias(
            self.load_winobias_data(winobias_path)
        )
        full_results['benchmark_results']['winobias'] = winobias_results
        
        # 4. SOTA comparison
        print("\n4. State-of-the-Art Comparison...")
        sota_comparison = self.compare_with_sota_on_benchmarks()
        full_results['comparison_with_sota'] = sota_comparison
        
        # 5. Generate summary
        summary = {
            'overall_bias_reduction': {
                'stereoset': stereoset_results['stereotype_mitigation_effectiveness'],
                'crows_pairs': crows_results['average_bias_reduction'],
                'winobias': winobias_results['gender_bias_reduction']
            },
            'overall_content_preservation': {
                'stereoset': stereoset_results['content_preservation'],
                'crows_pairs': crows_results['average_content_preservation'],
                'winobias': winobias_results['content_preservation']
            },
            'bias_detection_accuracy': {
                'stereoset': stereoset_results['bias_detection_accuracy'],
                'crows_pairs': crows_results['ranking_accuracy']
            }
        }
        
        # Calculate overall averages
        summary['average_bias_reduction'] = np.mean(list(summary['overall_bias_reduction'].values()))
        summary['average_content_preservation'] = np.mean(list(summary['overall_content_preservation'].values()))
        summary['average_detection_accuracy'] = np.mean(list(summary['bias_detection_accuracy'].values()))
        
        full_results['summary'] = summary
        
        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Performance:")
        print(f"  Average Bias Reduction: {summary['average_bias_reduction']:.1%}")
        print(f"  Average Content Preservation: {summary['average_content_preservation']:.1%}")
        print(f"  Average Detection Accuracy: {summary['average_detection_accuracy']:.1%}")
        
        print(f"\nBenchmark-Specific Results:")
        print(f"  StereoSet:")
        print(f"    - Bias Detection Accuracy: {stereoset_results['bias_detection_accuracy']:.1%}")
        print(f"    - Stereotype Mitigation: {stereoset_results['stereotype_mitigation_effectiveness']:.1%}")
        print(f"    - Content Preservation: {stereoset_results['content_preservation']:.1%}")
        
        print(f"  CrowS-Pairs:")
        print(f"    - Ranking Accuracy: {crows_results['ranking_accuracy']:.1%}")
        print(f"    - Average Bias Reduction: {crows_results['average_bias_reduction']:.1%}")
        print(f"    - Content Preservation: {crows_results['average_content_preservation']:.1%}")
        
        print(f"  WinoBias:")
        print(f"    - Gender Bias Reduction: {winobias_results['gender_bias_reduction']:.1%}")
        print(f"    - Stereotype Mitigation: {winobias_results['stereotype_mitigation']:.1%}")
        print(f"    - Content Preservation: {winobias_results['content_preservation']:.1%}")
        
        # SOTA comparison summary
        if 'graph_of_thought' in sota_comparison['metrics']['bias_reduction_effectiveness']:
            got_performance = sota_comparison['metrics']['bias_reduction_effectiveness']['graph_of_thought']['average']
            got_content = sota_comparison['metrics']['content_preservation']['graph_of_thought']['average']
            
            print(f"\nComparison with State-of-the-Art:")
            print(f"  Graph of Thought:")
            print(f"    - Average Bias Reduction: {got_performance:.1f}%")
            print(f"    - Average Content Retention: {got_content:.1%}")
            
            # Compare with best SOTA technique
            best_sota_technique = max(
                sota_comparison['metrics']['bias_reduction_effectiveness'].keys(),
                key=lambda x: sota_comparison['metrics']['bias_reduction_effectiveness'][x]['average']
                if x != 'graph_of_thought' else -1
            )
            
            if best_sota_technique != 'graph_of_thought':
                best_sota_performance = sota_comparison['metrics']['bias_reduction_effectiveness'][best_sota_technique]['average']
                improvement = got_performance - best_sota_performance
                print(f"  Best SOTA Technique ({best_sota_technique}): {best_sota_performance:.1f}%")
                print(f"  Improvement over best SOTA: {improvement:+.1f} percentage points")
        
        return full_results
    
    def export_results(self, filename: str = "benchmark_results.json"):
        """Export all benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results exported to {filename}")

# Function to replace in your original system
def replace_bias_detector_class():
    """
    REPLACE the BiasDetector class in your original code with EnhancedBiasDetector
    """
    pass

def replace_graph_of_thought_class():
    """
    REPLACE the GraphOfThought class in your original code with EnhancedGraphOfThought  
    """
    pass

def replace_bias_scores_dataclass():
    """
    REPLACE the BiasScores dataclass in your original code with EnhancedBiasScores
    """
    pass

def replace_graph_node_dataclass():
    """
    REPLACE the GraphNode dataclass in your original code with EnhancedGraphNode
    """
    pass

def replace_main_function():
    """
    REPLACE your main() function with this enhanced version that includes benchmark testing
    """
    def enhanced_main():
        # Initialize benchmark testing suite
        benchmark_suite = BenchmarkTestingSuite()
        
        # Run full benchmark evaluation
        results = benchmark_suite.run_full_benchmark_suite()
        
        # Export results
        benchmark_suite.export_results("enhanced_bias_mitigation_benchmark_results.json")
        
        return results
    
    return enhanced_main

# Usage example:
if __name__ == "__main__":
    # Run the enhanced benchmark testing
    suite = BenchmarkTestingSuite()
    results = suite.run_full_benchmark_suite()
    suite.export_results()