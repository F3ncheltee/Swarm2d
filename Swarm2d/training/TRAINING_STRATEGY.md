# Multi-Policy Swarm Training Strategy

## Overview

This document outlines the comprehensive training strategy for the 6-team swarm environment with three different policy architectures (MAAC, NCA, SharedActor) operating in both limited and global observation modes.

## Current State Analysis

### ✅ What's Working
- **Training Loop**: All 6 policies are instantiated and running
- **SAC Implementation**: Twin critics, target networks, alpha learning
- **Memory Systems**: NTMs, trail memories, persistent graph memory
- **Observation Generation**: Unified graphs, map processing, belief states
- **Contrastive Learning**: Integrated across all policies
- **Debug Infrastructure**: Comprehensive logging and monitoring

### 🔴 Performance Issues
- **Graph Generation Bottleneck**: 0.15-0.19s per step (80% of observation time)
- **Low FPS**: 0.5-0.6 FPS due to observation overhead
- **Training Limited**: Currently capped at 500 steps for profiling

### ❌ Missing Components
- **Hyperparameter Search**: Not implemented
- **Curriculum Learning**: Not integrated
- **Full Training Validation**: Updates not being executed properly

## Recommended Training Approach

### Phase 1: Immediate Fixes (Priority 1)

#### 1.1 Remove Training Limitations
- ✅ Remove 500-step limit
- ✅ Enable full training loop execution
- ✅ Fix update execution issues

#### 1.2 Performance Optimization
- **Graph Generation**: Optimize radius_graph construction
- **Observation Batching**: Improve memory efficiency
- **GPU Utilization**: Better tensor operations

#### 1.3 Debug Enhancement
- ✅ Add comprehensive update tracking
- ✅ Monitor gradient norms and losses
- ✅ Track buffer utilization

### Phase 2: Hyperparameter Search (Priority 2)

#### 2.1 Search Strategy
```python
# Use Bayesian optimization with Optuna
- MAAC: Focus on attention mechanisms, GNN layers
- NCA: Focus on NCA iterations, memory slots, belief dimensions
- SharedActor: Focus on trail memory, contrastive learning
```

#### 2.2 Search Parameters
- **Learning Rates**: 1e-5 to 1e-2 (log scale)
- **Architecture**: Hidden dimensions, layer counts, attention heads
- **Memory**: Slot counts, dimensions, update rates
- **Loss Coefficients**: Entropy, auxiliary, contrastive weights

#### 2.3 Evaluation Metrics
- **Primary**: Average episode reward
- **Secondary**: Training stability, convergence speed
- **Tertiary**: Memory efficiency, computational cost

### Phase 3: Curriculum Learning (Priority 3)

#### 3.1 Curriculum Stages
1. **Single Team** (No competition) - Learn basic behaviors
2. **Two Teams** (Light competition) - Learn cooperation
3. **Three Teams** (Moderate competition) - Learn strategy
4. **Full Six Teams** (Full competition) - Learn advanced tactics
5. **Dynamic Difficulty** (Adaptive) - Continuous challenge

#### 3.2 Progression Criteria
- Minimum episodes per stage: 50
- Performance threshold: 10.0 average reward
- Improvement rate: 5% per episode
- Patience: 10 episodes without improvement

### Phase 4: Advanced Training (Priority 4)

#### 4.1 Multi-Policy Coordination
- **Shared Experience**: Cross-policy learning
- **Role Specialization**: Adaptive role assignment
- **Team Coordination**: Inter-team communication

#### 4.2 Advanced Techniques
- **Self-Play**: Teams compete against each other
- **Population Training**: Multiple policy variants
- **Meta-Learning**: Fast adaptation to new scenarios

## Implementation Plan

### Week 1: Core Fixes
- [x] Remove training limitations
- [x] Add comprehensive debugging
- [ ] Optimize graph generation
- [ ] Fix update execution

### Week 2: Hyperparameter Search
- [ ] Implement Optuna integration
- [ ] Run initial search (20 trials per policy)
- [ ] Analyze results and select best parameters
- [ ] Validate with full training runs

### Week 3: Curriculum Learning
- [ ] Integrate curriculum manager
- [ ] Implement stage progression
- [ ] Test with simplified scenarios
- [ ] Validate performance improvements

### Week 4: Full Training
- [ ] Run complete training with best parameters
- [ ] Monitor convergence and stability
- [ ] Evaluate final performance
- [ ] Document results and insights

## Expected Outcomes

### Performance Targets
- **FPS**: >2.0 (4x improvement)
- **Training Speed**: <1 hour per episode
- **Convergence**: <500 episodes to plateau
- **Final Performance**: >50 average reward per team

### Success Metrics
- **Stability**: No gradient explosions or NaN values
- **Convergence**: Smooth learning curves
- **Efficiency**: Optimal resource utilization
- **Robustness**: Consistent performance across runs

## Risk Mitigation

### Technical Risks
- **Memory Issues**: Monitor GPU memory usage
- **Gradient Problems**: Implement gradient clipping
- **Convergence Issues**: Use learning rate scheduling

### Training Risks
- **Overfitting**: Use validation episodes
- **Catastrophic Forgetting**: Implement experience replay
- **Mode Collapse**: Use diverse exploration strategies

## Monitoring and Evaluation

### Real-time Monitoring
- **TensorBoard**: Loss curves, reward trends, gradient norms
- **Console Output**: Episode summaries, update statistics
- **System Metrics**: GPU utilization, memory usage

### Periodic Evaluation
- **Every 50 Episodes**: Full evaluation phase
- **Every 100 Episodes**: Hyperparameter adjustment
- **Every 200 Episodes**: Curriculum progression check

### Final Evaluation
- **Performance**: Average reward, win rate, efficiency
- **Robustness**: Performance across different seeds
- **Scalability**: Performance with different team sizes

## Conclusion

This strategy provides a systematic approach to training the multi-policy swarm system. By addressing immediate issues first, then implementing advanced techniques, we can achieve robust and efficient training while maintaining system stability.

The key is to start simple, validate each component, and gradually increase complexity as the system proves stable and effective.

