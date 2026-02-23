from typing import Dict, List, Optional, Tuple
from itertools import combinations
import numpy as np
import torch
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool
from alphagen.data.expression import Expression, expression_edit_distance
from alphagen_qlib.stock_data import StockData
from alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph, ExpressionNode
from alphagen.utils.correlation import batch_pearsonr
from sentence_transformers import SentenceTransformer

class AlphaKnowledgePool(AlphaPool):
    def __init__(
            self,
            capacity: int,
            stock_data: StockData,
            target: Expression,
            ic_mut_threshold: float = 0.9,
            top_k : int = 15,
            depth_decay: float = 0.05, 
            embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
            use_res_correlation: bool = True,
            use_semantic_similarity: bool = True,
            use_edit_distance: bool = False,
            separate_leaf_non_leaf: bool = False,
            knowledge_graph: Optional[ExpressionKnowledgeGraph] = None,
            times_decay: float = 0.1,
            start_times: int = 2,
            device: str = "cuda:0"
    ):
        super().__init__(capacity, stock_data, target)
        self.ic_mut_threshold = ic_mut_threshold
        self.topics: List[Optional[str]] = [None] * (capacity + 1)
        self.descriptions: List[Optional[str]] = [None] * (capacity + 1)
        self.expr2node: List[Optional[ExpressionNode]] = [None] * (capacity + 1)
        self.icir: List[Optional[float]] = [None] * (capacity + 1)
        self.depth_decay = depth_decay
        self.times_decay = times_decay
        self.start_times = start_times
    #     self.embedding_model = SentenceTransformer(embedding_model_name, model_kwargs={"device_map": device},
    # tokenizer_kwargs={"padding_side": "left"})

        self.use_res_correlation = use_res_correlation
        if use_semantic_similarity:
            self.embedding_model = SentenceTransformer(embedding_model_name, model_kwargs={"device_map": device},
    tokenizer_kwargs={"padding_side": "left"})
            self.use_semantic_similarity = use_semantic_similarity
        else:
            self.embedding_model = None
            self.use_semantic_similarity = False
        
        self.use_edit_distance = use_edit_distance
        self.separate_leaf_non_leaf = separate_leaf_non_leaf
        self.knowledge_graph = knowledge_graph
        self.top_k = top_k


    def attach_knowledge_graph(self, graph: ExpressionKnowledgeGraph) -> None:
        """Attach an expression knowledge graph for node lookup during search."""
        self.knowledge_graph = graph

    def register_expression_node(self, node: ExpressionNode) -> None:
        """Update internal mapping for a factor now represented in the knowledge graph."""
        expr_source = node.payload.source
        for idx in range(self.size):
            expr = self.exprs[idx]
            if expr is not None and str(expr) == expr_source:
                self.expr2node[idx] = node
                break


    def get_ic(self, expr: Expression | str) -> Optional[float]:
        if isinstance(expr, str):
            expr = eval(expr)
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99, expr=expr)
        return ic_ret
    
    def get_ic_icir_mutics(self, expr: Expression | str) -> Optional[Tuple[float, float, List[float]]]:
        if isinstance(expr, str):
            expr = eval(expr)
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic, icir, mutics = self._calc_ics_and_icir(value)
        return ic, icir, mutics

    def get_mutl_ic(self, expr: Expression | str) -> Optional[float]:
        if isinstance(expr, str):
            expr = eval(expr)
        assert isinstance(expr, Expression)
        value  = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None:
            return None
        return np.max(np.abs(ic_mut))
    
    def search(self,) -> Tuple[List[ExpressionNode], List[Expression]]:
        def stable_sigmoid(x: np.ndarray) -> np.ndarray:
            """Numerically stable sigmoid for numpy arrays."""
            return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        def normalize_scores(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values
            standardized = (values - values.mean()) / (values.std() + 1e-6)
            return stable_sigmoid(standardized)

        def clip_prob(value: float) -> float:
            return float(np.clip(value, 1e-6, 1 - 1e-6))

        def lookup_node(expr_source: str) -> Optional[ExpressionNode]:
            if self.knowledge_graph is None:
                return None
            return self.knowledge_graph.search(expr_source)

        if self.size == 0:
            return []

        candidate_pairs: List[Tuple[int, ExpressionNode]] = []
        for pool_idx in range(self.size):
            expr = self.exprs[pool_idx]
            if expr is None:
                continue
            node = self.expr2node[pool_idx]
            # if node is None:
            #     node = lookup_node(str(expr))
            #     if node is not None:
            #         self.expr2node[pool_idx] = node
            if node is None:
                continue
            candidate_pairs.append((pool_idx, node))

        if not candidate_pairs:
            return []

        valid_indices = [idx for idx, _ in candidate_pairs]
        nodes = [node for _, node in candidate_pairs]
        pool_idxs = [pool_idx for pool_idx, _ in candidate_pairs]

        icir_raw = np.array([
            self.icir[idx] if self.icir[idx] is not None else node.icir
            for idx, node in candidate_pairs
        ], dtype=float)

        finite_mask = np.isfinite(icir_raw)
        mean_icir = icir_raw[finite_mask].mean() if finite_mask.any() else 0.0
        std_icir = icir_raw[finite_mask].std() if finite_mask.any() else 0.0
        icir_logit = stable_sigmoid((icir_raw - mean_icir) / (std_icir + 1e-6))

        depths = np.array([node.depth for node in nodes], dtype=float)
        times = np.array([node.times for node in nodes], dtype=float)
        depth_decays = np.power(max(1 - self.depth_decay, 1e-6), depths)
        times_decays = np.power(max(1 - self.times_decay, 1e-6), np.maximum(times - self.start_times, 0))
        quality_prob = np.clip(icir_logit * depth_decays * times_decays, 1e-6, 1 - 1e-6)
        # quality_prob = normalize_scores(quality_prob)

        corr_matrix = np.abs(self.mutual_ics[np.ix_(valid_indices, valid_indices)])
        np.fill_diagonal(corr_matrix, 0.0)

        semantic_matrix: Optional[np.ndarray] = None
        if self.use_semantic_similarity and getattr(self, "embedding_model", None) is not None:
            texts: List[str] = []
            for idx, node in candidate_pairs:
                text = self.descriptions[idx] or node.topic or node.description or node.payload.source
                texts.append(text)
            embeddings = self.embedding_model.encode(  # type: ignore[union-attr]
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            if hasattr(embeddings, "to"):
                embeddings = embeddings.to("cpu").float()
            semantic_matrix = (embeddings @ embeddings.T).clamp(-1.0, 1.0).numpy()

        expr_strings = [node.payload.source for node in nodes]
        expr_to_position = {expr: pos for pos, expr in enumerate(expr_strings)}

        num_nodes = len(nodes)
        expr_lengths = np.array([node.payload.length for node in nodes], dtype=float)

        edit_matrix: Optional[np.ndarray] = None
        if self.use_edit_distance:
            edit_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = expression_edit_distance(
                        nodes[i].payload.expression,
                        nodes[j].payload.expression,
                    )
                    length_sum = expr_lengths[i] + expr_lengths[j]
                    ratio = dist / (length_sum + 1e-6)
                    edit_matrix[i, j] = edit_matrix[j, i] = clip_prob(ratio)

        node_values: List[Tensor] = []
        for pos, (pool_idx, node) in enumerate(candidate_pairs):
            value = self.values[pool_idx]
            if value is None:
                value = self._normalize_by_day(node.payload.expression.evaluate(self.data))
            node_values.append(value)

        value_cache: Dict[str, Tensor] = {node.payload.source: node_values[pos] for pos, node in enumerate(nodes)}

        def get_factor_value(node: ExpressionNode) -> Optional[Tensor]:
            key = node.payload.source
            if key in value_cache:
                return value_cache[key]
            try:
                value_cache[key] = self._normalize_by_day(node.payload.expression.evaluate(self.data))
            except Exception:
                return None
            return value_cache[key]

        pool_quality_raw = np.ones(num_nodes, dtype=float)
        group_labels = np.array(["leaf"] * num_nodes, dtype=object)

        def compute_leaf_quality(pos: int) -> float:
            corr_score = 1.0
            if self.use_res_correlation:
                others = [j for j in range(num_nodes) if j != pos]
                if others:
                    corr_vals = corr_matrix[pos, others]
                    corr_score = clip_prob(1.0 - float(np.mean(corr_vals)))
            semantic_score = 1.0
            if semantic_matrix is not None:
                others = [j for j in range(num_nodes) if j != pos]
                if others:
                    sims = semantic_matrix[pos, others]
                    mapped = ((np.clip(sims, -1.0, 1.0) + 1.0) * 0.5).mean()
                    semantic_score = clip_prob(1.0 - float(mapped))
            edit_score = 1.0
            if edit_matrix is not None:
                others = [j for j in range(num_nodes) if j != pos]
                if others:
                    edit_score = clip_prob(float(edit_matrix[pos, others].mean()))
            return clip_prob(corr_score * semantic_score * edit_score)

        def compute_non_leaf_quality(pos: int, pool_idx: int, node: ExpressionNode) -> float:
            children_info: List[Tuple[ExpressionNode, Optional[int]]] = []
            for child_expr in node.children:
                expr_key = child_expr.strip()
                child_pos = expr_to_position.get(expr_key)
                child_node = nodes[child_pos] if child_pos is not None else lookup_node(expr_key)
                if child_node is not None:
                    children_info.append((child_node, child_pos))
            if not children_info:
                return compute_leaf_quality(pos)

            parent_icir = abs(icir_raw[pos])
            denom = max(parent_icir, 1e-6)
            child_icirs = []
            for child_node, child_pos in children_info:
                if child_pos is not None:
                    child_icirs.append(abs(icir_raw[child_pos]))
                else:
                    child_icirs.append(abs(child_node.icir))
            child_icirs = np.array(child_icirs, dtype=float)
            percent_gains = (child_icirs - parent_icir) / denom
            gain_scores = normalize_scores(percent_gains)
            mean_gain = clip_prob(float(gain_scores.mean()))

            parent_value = node_values[pos]
            parent_child_corrs: List[float] = []
            child_values: List[Tensor] = []
            for child_node, child_pos in children_info:
                if child_pos is not None:
                    corr_val = float(np.clip(corr_matrix[pos, child_pos], 0.0, 1.0))
                    child_values.append(node_values[child_pos])
                    parent_child_corrs.append(corr_val)
                else:
                    child_value = get_factor_value(child_node)
                    if child_value is None:
                        continue
                    corr_tensor = batch_pearsonr(parent_value, child_value).mean()
                    corr_val = float(torch.abs(corr_tensor).item())
                    corr_val = float(np.clip(corr_val, 0.0, 1.0))
                    child_values.append(child_value)
                    parent_child_corrs.append(corr_val)

            parent_child_avg = float(np.mean(parent_child_corrs)) if parent_child_corrs else 0.0
            parent_child_avg = float(np.clip(parent_child_avg, 0.0, 1.0))

            sibling_corrs: List[float] = []
            if len(child_values) > 1:
                for i in range(len(child_values)):
                    for j in range(i + 1, len(child_values)):
                        corr_tensor = batch_pearsonr(child_values[i], child_values[j]).mean()
                        corr_val = float(torch.abs(corr_tensor).item())
                        sibling_corrs.append(float(np.clip(corr_val, 0.0, 1.0)))
            child_child_avg = float(np.mean(sibling_corrs)) if sibling_corrs else 0.0
            child_child_avg = float(np.clip(child_child_avg, 0.0, 1.0))

            sparsity_score = clip_prob(1.0 - parent_child_avg * child_child_avg)
            return clip_prob(mean_gain * sparsity_score)

        for pos, (pool_idx, node) in enumerate(candidate_pairs):
            if node.children:
                pool_quality_raw[pos] = compute_non_leaf_quality(pos, pool_idx, node)
                group_labels[pos] = "non_leaf"
            else:
                pool_quality_raw[pos] = compute_leaf_quality(pos)

        pool_quality_combined = normalize_scores(pool_quality_raw.copy())
        pool_quality_split = pool_quality_combined.copy()

        leaf_positions = [i for i, label in enumerate(group_labels) if label == "leaf"]
        non_leaf_positions = [i for i, label in enumerate(group_labels) if label == "non_leaf"]

        # if leaf_positions:
        #     pool_quality_split[leaf_positions] = normalize_scores(pool_quality_raw[leaf_positions])
        # if non_leaf_positions:
        #     pool_quality_split[non_leaf_positions] = normalize_scores(pool_quality_raw[non_leaf_positions])

        pool_quality_combined = np.clip(pool_quality_combined, 1e-6, 1 - 1e-6)
        pool_quality_split = np.clip(pool_quality_split, 1e-6, 1 - 1e-6)

        scores_combined = np.clip(quality_prob * pool_quality_combined, 1e-6, 1 - 1e-6)
        scores_split = np.clip(quality_prob * pool_quality_split, 1e-6, 1 - 1e-6)

        if not self.separate_leaf_non_leaf:
            ranking = np.argsort(-scores_combined)
            top_positions = ranking[: min(self.top_k, len(ranking))]
            return [nodes[pos] for pos in top_positions], [self.exprs[pool_idxs[pos]] for pos in top_positions]

        half_k = max(1, self.top_k // 2)
        selected_positions: List[int] = []

        leaf_sorted = sorted(leaf_positions, key=lambda p: scores_split[p], reverse=True)
        non_leaf_sorted = sorted(non_leaf_positions, key=lambda p: scores_split[p], reverse=True)

        selected_positions.extend(leaf_sorted[:min(half_k, len(leaf_sorted))])
        selected_positions.extend(non_leaf_sorted[:min(half_k, len(non_leaf_sorted))])

        selected_positions = list(dict.fromkeys(selected_positions))

        if len(selected_positions) < self.top_k:
            remaining_candidates = [
                pos for pos in leaf_sorted[min(half_k, len(leaf_sorted)):] +
                non_leaf_sorted[min(half_k, len(non_leaf_sorted)):] if pos not in selected_positions
            ]
            remaining_sorted = sorted(remaining_candidates, key=lambda p: scores_split[p], reverse=True)
            selected_positions.extend(remaining_sorted[: self.top_k - len(selected_positions)])

        selected_positions = selected_positions[: self.top_k]
        selected_positions.sort(key=lambda p: scores_split[p], reverse=True)
        for pos in selected_positions:
            nodes[pos].times += 1
        return [nodes[pos] for pos in selected_positions], [self.exprs[pool_idxs[pos]] for pos in selected_positions]

    def try_new_expr(self, expr: Expression | str, topic: str, description: str, expression_node: Optional[ExpressionNode] = None) -> bool:
        if isinstance(expr, str):
            expr = eval(expr)
        assert isinstance(expr, Expression)
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, icir, ic_mut = self._calc_ics_and_icir(value, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None:
            return False
        ic_ret = np.abs(ic_ret)
        ic_mut = np.abs(ic_mut)

        # if expression_node is None and self.knowledge_graph is not None:
        #     expression_node = self.knowledge_graph.search(str(expr))

        if self.size < self.capacity:
            if ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold:
                self._add_factor(expr, value, ic_ret, icir, ic_mut, topic, description, expression_node)
                print(f"[Pool Add] {expr}")
                return True
            print(f"[Pool Reject] {expr}")
            return False

        min_ic_idx = np.argmin(self.single_ics[:self.size])
        min_ic = self.single_ics[min_ic_idx]

        if ic_ret > min_ic and (ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold):
            self._add_factor(expr, value, ic_ret, icir, ic_mut, topic, description, expression_node)
            print(f"[Pool Add] {expr}")
            print(f"[Pool Pop] {self.exprs[np.argmin(self.single_ics[:self.size])]}")
            self._pop()
            return True
        print(f"[Pool Reject] {expr}")
        return False

    def _add_factor(
        self,
        expr: Expression,
        value: Tensor,
        ic_ret: float,
        ic_ir: float,
        ic_mut: List[float],
        topic: str,
        description: str,
        expression_node: Optional[ExpressionNode] = None
    ):
        super()._add_factor(expr, value, ic_ret, ic_mut)
        n = self.size - 1  # size was incremented in parent method
        self.topics[n] = topic
        self.descriptions[n] = description
        # if expression_node is None and self.knowledge_graph is not None:
        #     expression_node = self.knowledge_graph.search(str(expr))
        self.expr2node[n] = expression_node
        self.icir[n] = ic_ir
    

    def _pop(self) -> None:
        # Pop the factor with the lowest ic
        if self.size <= self.capacity:
            return
        idx = np.argmin(self.single_ics[:self.size])
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity
    
    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        # Call parent method to handle standard factor swapping
        super()._swap_idx(i, j)
        # Swap embeddings
        self.topics[i], self.topics[j] = self.topics[j], self.topics[i]
        self.descriptions[i], self.descriptions[j] = self.descriptions[j], self.descriptions[i]   
        self.expr2node[i], self.expr2node[j] = self.expr2node[j], self.expr2node[i]
        self.icir[i], self.icir[j] = self.icir[j], self.icir[i]