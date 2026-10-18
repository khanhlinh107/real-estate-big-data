import unittest
from unittest.mock import patch

import pandas as pd

from web_app import app as search_app


class PropertySearchTests(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame(
            [
                {
                    "id": 13,
                    "name_nsi": "Bratislava",
                    "price": 250000,
                    "area": 70,
                    "rooms": 2,
                    "district": "Nové Zámky",
                    "type": "2-room apartment",
                    "condition": "Original condition",
                    "environment": 8.5,
                    "floor": 3,
                },
                {
                    "id": 2,
                    "name_nsi": "Štúrovo",
                    "price": 300000,
                    "area": 130,
                    "rooms": 3,
                    "district": "Košice",
                    "type": "3-room apartment",
                    "condition": "Partial reconstruction",
                    "environment": 9.3,
                    "floor": 2,
                },
                {
                    "id": 3,
                    "name_nsi": "Semerovo",
                    "price": 250000,
                    "area": 70,
                    "rooms": 2,
                    "district": "Nové Zámky",
                    "type": "2-room apartment",
                    "condition": "Original condition",
                    "environment": 8.5,
                    "floor": 3,
                },
            ]
        )

    def _matching_ids(self, query, field=None):
        results = search_app.build_filter(self.records, query=query, field=field)
        return results["id"].tolist()

    def test_partial_area_name_matches(self):
        self.assertEqual(self._matching_ids("Brat"), [13])

    def test_property_type_matches_vietnamese_with_diacritics(self):
        self.assertEqual(self._matching_ids("Căn hộ 3 phòng", "type"), [2])

    def test_property_type_matches_vietnamese_without_diacritics(self):
        self.assertEqual(self._matching_ids("can ho 3 phong", "type"), [2])

    def test_condition_matches_vietnamese_with_diacritics(self):
        self.assertEqual(self._matching_ids("Đã cải tạo một phần", "condition"), [2])

    def test_condition_matches_vietnamese_without_diacritics(self):
        self.assertEqual(self._matching_ids("da cai tao mot phan", "condition"), [2])

    def test_price_area_and_room_search(self):
        self.assertEqual(self._matching_ids("300000", "price"), [2])
        self.assertEqual(self._matching_ids("300,000", "price"), [2])
        self.assertEqual(self._matching_ids("130", "area"), [2])
        self.assertEqual(self._matching_ids("2", "rooms"), [13, 3])

    def test_decimal_search_does_not_remove_decimal_point(self):
        record = self.records.iloc[[0]].copy()
        record["price"] = record["price"].astype(float)
        record.loc[:, "price"] = 20.0
        self.assertEqual(
            search_app.build_filter(record, query="20.0", field="price")["id"].tolist(),
            [13],
        )
        self.assertTrue(search_app.build_filter(record, query="2.0", field="price").empty)

    def test_numeric_query_only_returns_records_with_visible_match(self):
        self.assertEqual(self._matching_ids("3"), [2])

    def test_internal_id_and_unshown_metadata_are_not_searchable(self):
        internal_only_record = self.records[self.records["id"] == 3]
        self.assertTrue(search_app.build_filter(internal_only_record, query="3").empty)
        self.assertTrue(search_app.build_filter(internal_only_record, query="3", field="id").empty)
        floor_only_record = self.records[self.records["id"] == 13]
        self.assertTrue(search_app.build_filter(floor_only_record, query="3").empty)

    def test_whitespace_only_query_returns_all_records(self):
        self.assertEqual(self._matching_ids("   "), [13, 2, 3])

    def test_no_results_message_is_rendered(self):
        with patch.object(search_app, "load_dataset", return_value=self.records), patch.object(
            search_app, "get_dataset_snapshot", return_value={"total_listings": 3}
        ):
            response = search_app.app.test_client().get("/data-explorer?q=not-found")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Không tìm thấy kết quả phù hợp".encode(), response.data)

    def test_suggestions_use_display_values_and_exclude_nonmatching_fields(self):
        with patch.object(search_app, "load_dataset", return_value=self.records):
            response = search_app.app.test_client().get("/api/search?q=can%20ho%203%20phong")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["suggestions"],
            [
                {
                    "value": "Căn hộ 3 phòng",
                    "field": "type",
                    "field_label": "Loại bất động sản",
                    "count": 1,
                }
            ],
        )

    def test_api_suggestions_refresh_for_changed_queries(self):
        with patch.object(search_app, "load_dataset", return_value=self.records):
            client = search_app.app.test_client()
            first = client.get("/api/search?q=Brat").get_json()["suggestions"]
            second = client.get("/api/search?q=Ko%C5%A1").get_json()["suggestions"]
        self.assertTrue(any(item["value"] == "Bratislava" for item in first))
        self.assertTrue(any(item["value"] == "Košice" for item in second))
        self.assertFalse(any(item["value"] == "Bratislava" for item in second))


if __name__ == "__main__":
    unittest.main()
