from iot_security.cuckoo_filter import CuckooFilter


def test_insert_lookup_delete_cycle():
    cf = CuckooFilter(capacity=200, bucket_size=4, max_kicks=100)
    signature = "12.3_6_0"

    assert cf.insert(signature) is True
    assert cf.lookup(signature) is True
    assert cf.delete(signature) is True
    assert cf.lookup(signature) is False
