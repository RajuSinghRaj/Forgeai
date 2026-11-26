# Ship Spares Corpus Synthesis Instructions

Generate a synthetic, internally consistent ship-spares inventory catalogue and supplier technical corpus for demonstration purposes:

1. `inventory_catalogue.csv` — 18 spare-part SKUs with `item_id`, `description`, `category`, `stock_qty`, `unit_price`, `currency`, `warehouse_location`, `supplier_id`, `supplier_name`, `part_number`, and `stock_status_note`.
   - Include in-stock items, low-stock items, zero-stock items, and one item (`VA-900`) with a blank/TBD `unit_price` (`RFQ_REQUIRED`) to test refusal against inventing prices.
   - Include similar-sounding descriptions (`MV-101` Main Engine Cooling Pump Mechanical Seal vs. `MV-102` Auxiliary Bilge Pump Mechanical Seal; `FL-201` 10-Micron Fuel Oil Filter vs. `FL-202` 25-Micron Lube Oil Filter) to test retrieval precision.
2. `supplier_a_nordic_marine_specs.pdf` — Synthetic technical & commercial specification sheet for `SUP-NORDIC` (Nordic Marine Spares GmbH) covering seal/pump/valve specs, MOQ rules, air vs. sea freight lead times, Incoterms, and explicit distinction between supplier availability and placed POs.
3. `supplier_b_pacific_propulsion_specs.pdf` — Synthetic technical & commercial specification sheet for `SUP-PACIFIC` (Pacific Propulsion Parts Ltd) covering filter/injector/sensor specs, MOQ rules, compatibility warnings (10-micron vs. 25-micron filters), Singapore dispatch lead times, and RFQ-only pricing notice for custom assembly `VA-900`.

Label all files clearly as synthetic demonstration data.

