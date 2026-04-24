import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/rel_MSG')

from models.scene_graph import (
    PairProposalNetwork,
    RelationHead,
    SceneGraphHead,
    SceneGraphModule,
    TripletGenerator,
    SceneGraphEvaluator,
    prepare_gt_importance_matrix,
)


def test_pair_proposal_network():
    print("=" * 60)
    print("Testing PairProposalNetwork")
    print("=" * 60)

    batch_size = 2
    num_obj = 10
    feat_channels = 256
    num_classes = 133

    model = PairProposalNetwork(
        num_obj_query=num_obj,
        num_rel_query=5,
        feat_channels=feat_channels,
        num_classes=num_classes,
        use_matrix_learner=True,
    )

    object_query = torch.randn(batch_size, num_obj, feat_channels)
    object_cls = torch.randn(batch_size, num_obj, num_classes + 1)
    object_cls = torch.softmax(object_cls, dim=-1)

    with torch.no_grad():
        outputs = model(object_query, object_cls)

    print(f"Input object_query shape: {object_query.shape}")
    print(f"Output keys: {outputs.keys()}")
    print(f"importance shape: {outputs['importance'].shape}")
    print(f"sub_pos shape: {outputs['sub_pos'].shape}")
    print(f"obj_pos shape: {outputs['obj_pos'].shape}")
    print(f"sub_query shape: {outputs['sub_query'].shape}")
    print(f"obj_query shape: {outputs['obj_query'].shape}")
    print(f"pair_query shape: {outputs['pair_query'].shape}")
    print(f"num_selected_pairs: {outputs['num_selected_pairs']}")

    assert outputs['importance'].shape == (batch_size, num_obj, num_obj)
    assert outputs['sub_pos'].shape == (batch_size, outputs['num_selected_pairs'])
    assert outputs['obj_pos'].shape == (batch_size, outputs['num_selected_pairs'])
    print("PairProposalNetwork test PASSED!")
    return outputs


def test_relation_head():
    print("\n" + "=" * 60)
    print("Testing RelationHead")
    print("=" * 60)

    batch_size = 2
    num_rel = 5
    feat_channels = 256
    num_relations = 56

    model = RelationHead(
        num_rel_query=num_rel,
        num_relations=num_relations,
        feat_channels=feat_channels,
        num_rel_decoder_layers=2,
    )

    sub_query = torch.randn(batch_size, num_rel, feat_channels)
    obj_query = torch.randn(batch_size, num_rel, feat_channels)

    with torch.no_grad():
        outputs = model(sub_query, obj_query)

    print(f"Input sub_query shape: {sub_query.shape}")
    print(f"Input obj_query shape: {obj_query.shape}")
    print(f"Output keys: {outputs.keys()}")
    print(f"rel_pred shape: {outputs['rel_pred'].shape}")

    assert outputs['rel_pred'].shape == (batch_size, num_rel, num_relations)
    print("RelationHead test PASSED!")
    return outputs


def test_scene_graph_head():
    print("\n" + "=" * 60)
    print("Testing SceneGraphHead")
    print("=" * 60)

    batch_size = 2
    num_obj = 10
    num_rel = 5
    feat_channels = 256
    num_classes = 133
    num_relations = 56

    model = SceneGraphHead(
        num_classes=num_classes,
        num_relations=num_relations,
        num_obj_query=num_obj,
        num_rel_query=num_rel,
        feat_channels=feat_channels,
    )

    object_query = torch.randn(batch_size, num_obj, feat_channels)
    object_cls = torch.randn(batch_size, num_obj, num_classes + 1)
    object_cls = torch.softmax(object_cls, dim=-1)

    with torch.no_grad():
        outputs = model(object_query, object_cls, mode='sgdet')

    print(f"Input object_query shape: {object_query.shape}")
    print(f"Output keys: {outputs.keys()}")
    print(f"triplets type: {type(outputs['triplets'])}")
    print(f"Number of triplets: {len(outputs['triplets'])}")
    if len(outputs['triplets']) > 0:
        print(f"First triplet: {outputs['triplets'][0]}")

    assert 'triplets' in outputs
    assert 'rel_pred' in outputs
    print("SceneGraphHead test PASSED!")
    return outputs


def test_scene_graph_module():
    print("\n" + "=" * 60)
    print("Testing SceneGraphModule (full pipeline)")
    print("=" * 60)

    batch_size = 2
    num_obj = 10
    num_rel = 5
    embed_dims = 1024
    feat_channels = 256
    num_classes = 133
    num_relations = 56

    model = SceneGraphModule(
        num_classes=num_classes,
        num_relations=num_relations,
        num_obj_query=num_obj,
        num_rel_query=num_rel,
        feat_channels=feat_channels,
        embed_dims=embed_dims,
    )

    object_query = torch.randn(batch_size, num_obj, embed_dims)
    object_cls = torch.randn(batch_size, num_obj, num_classes + 1)
    object_cls = torch.softmax(object_cls, dim=-1)

    with torch.no_grad():
        outputs = model(object_query, object_cls, mode='sgdet')

    print(f"Input object_query shape: {object_query.shape}")
    print(f"Output keys: {outputs.keys()}")
    print(f"rel_pred shape: {outputs['rel_pred'].shape}")
    print(f"Number of triplets: {len(outputs['triplets'])}")

    assert outputs['rel_pred'].shape == (batch_size, num_rel, num_relations)
    print("SceneGraphModule test PASSED!")
    return outputs


def test_loss_computation():
    print("\n" + "=" * 60)
    print("Testing Loss Computation")
    print("=" * 60)

    batch_size = 2
    num_obj = 10
    num_rel = 5
    num_rel_actual = 3
    feat_channels = 256
    num_classes = 133
    num_relations = 56

    model = SceneGraphHead(
        num_classes=num_classes,
        num_relations=num_relations,
        num_obj_query=num_obj,
        num_rel_query=num_rel,
        feat_channels=feat_channels,
    )

    object_query = torch.randn(batch_size, num_obj, feat_channels)
    object_cls = torch.randn(batch_size, num_obj, num_classes + 1)
    object_cls = torch.softmax(object_cls, dim=-1)

    outputs = model(object_query, object_cls, mode='sgdet')

    gt_rels = torch.randint(0, num_obj, (batch_size, num_rel_actual, 3))
    gt_importance = prepare_gt_importance_matrix(
        gt_rels.view(-1, 3), num_obj, object_query.device
    ).unsqueeze(0).expand(batch_size, -1, -1)
    gt_sub_labels = gt_rels[:, :, 0]
    gt_obj_labels = gt_rels[:, :, 1]

    loss_dict = model.compute_loss(
        outputs=outputs,
        gt_rels=gt_rels,
        gt_importance=gt_importance,
        gt_sub_labels=gt_sub_labels,
        gt_obj_labels=gt_obj_labels,
    )

    print(f"Loss keys: {loss_dict.keys()}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    assert 'loss_match' in loss_dict or 'loss_rel_cls' in loss_dict
    print("Loss computation test PASSED!")
    return loss_dict


def test_triplet_generator():
    print("\n" + "=" * 60)
    print("Testing TripletGenerator")
    print("=" * 60)

    batch_size = 2
    num_rel = 10
    num_classes = 133
    num_relations = 56

    generator = TripletGenerator(
        num_relations=num_relations,
        num_classes=num_classes,
    )

    sub_pos = torch.randint(0, 10, (batch_size, num_rel))
    obj_pos = torch.randint(0, 10, (batch_size, num_rel))
    rel_pred = torch.randn(batch_size, num_rel, num_relations)
    sub_cls = torch.randn(batch_size, num_rel, num_classes + 1)
    sub_cls = torch.softmax(sub_cls, dim=-1)
    obj_cls = torch.randn(batch_size, num_rel, num_classes + 1)
    obj_cls = torch.softmax(obj_cls, dim=-1)

    triplets = generator(
        sub_pos=sub_pos,
        obj_pos=obj_pos,
        rel_pred=rel_pred,
        sub_cls=sub_cls,
        obj_cls=obj_cls,
        mode='sgdet',
    )

    print(f"Number of triplets generated: {len(triplets)}")
    if len(triplets) > 0:
        print(f"First triplet: {triplets[0]}")

    assert len(triplets) == batch_size * num_rel
    print("TripletGenerator test PASSED!")
    return triplets


def test_scene_graph_evaluator():
    print("\n" + "=" * 60)
    print("Testing SceneGraphEvaluator")
    print("=" * 60)

    num_samples = 2
    num_obj = 5
    num_pred = 3
    num_relations = 56

    groundtruths = []
    predictions = []

    for i in range(num_samples):
        gt_rels = torch.randint(0, num_obj, (num_pred, 2))
        gt_rels = torch.cat([gt_rels, torch.randint(0, num_relations, (num_pred, 1))], dim=1)
        gt_labels = torch.randint(0, 133, (num_obj,))

        pred_rels = torch.randint(0, num_obj, (num_pred, 2))
        rel_scores = torch.randn(num_pred, num_relations)
        rel_scores = torch.softmax(rel_scores, dim=-1)
        pred_labels = torch.randint(0, 133, (num_obj,))
        pred_boxes = torch.randn(num_obj, 5)

        groundtruths.append({
            'rels': gt_rels,
            'classes': gt_labels,
            'boxes': torch.randn(num_obj, 4),
        })

        predictions.append({
            'rel_pair_idxes': pred_rels,
            'rel_dists': rel_scores,
            'labels': pred_labels,
            'refine_bboxes': pred_boxes,
        })

    evaluator = SceneGraphEvaluator(
        mode='sgdet',
        num_predicates=num_relations,
        iou_thresh=0.5,
        detection_method='bbox',
    )

    for gt, pred in zip(groundtruths, predictions):
        evaluator.update(
            gt_rels=gt['rels'],
            pred_rels=pred['rel_pair_idxes'],
            gt_classes=gt['classes'],
            pred_classes=pred['labels'],
            gt_boxes=gt['boxes'],
            pred_boxes=pred['refine_bboxes'],
            rel_scores=pred['rel_dists'],
        )

    results = evaluator.compute()

    print(f"Evaluation results: {results}")

    assert 'sgdet_recall_20' in results or 'sgdet_recall_50' in results or 'sgdet_recall_100' in results
    print("SceneGraphEvaluator test PASSED!")
    return results


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SCENE GRAPH MODULE VALIDATION TESTS")
    print("=" * 60)

    test_pair_proposal_network()
    test_relation_head()
    test_scene_graph_head()
    test_scene_graph_module()
    test_loss_computation()
    test_triplet_generator()
    test_scene_graph_evaluator()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
