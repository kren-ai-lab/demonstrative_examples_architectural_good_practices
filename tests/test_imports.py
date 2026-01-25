def test_public_imports():
    import mcs
    from mcs.schemas import DatasetSpec, SplitSpec, EmbeddingSpec, TrainSpec, ExecutionSpec
    from mcs.validation import validate_all_5specs
    from mcs.provenance import ProvenanceRecord
    from mcs.registry import LocalRegistry

    assert callable(mcs.get_mcs_version)
    assert callable(mcs.run_pack)
    assert DatasetSpec and SplitSpec and EmbeddingSpec and TrainSpec and ExecutionSpec
    assert callable(validate_all_5specs)
    assert ProvenanceRecord
    assert LocalRegistry
