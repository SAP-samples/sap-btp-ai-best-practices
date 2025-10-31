## Preprocessing Steps (Formal Specification)

This document formalizes every transformation and check performed in `ui/data_preprocessing.py` using mathematical notation. Column names are referenced in backticks; equations use month/day/year parsing for dates.

### 1) Ingestion and Basic Parsing
- Load all CSV files in the chosen directory; vertically concatenate them.
- Date parsing for columns: `Requested delivery date`, `Sales Document Created Date`, `Actual GI Date`, `Invoice Creation Date`, `Shelf Life Expiration Date`, `Original Requested Delivery Date`, `Header Pricing Date`, `Item Pricing Date`, `Batch Manufacture Date` as
  $$ 
  D \leftarrow \text{to\_datetime}(\text{value}, \text{format} = \%m/\%d/\%y, \text{errors} = \text{"coerce"}) 
  $$
- Time parsing for `Entry time` to time-of-day.
- Numeric parsing for columns like `Sales Order item qty`, `Order item value`, `Unit Price`, etc., removing thousands separators and coercing to numeric.

### 2) Filtering Rows
- Let each row have `BillingStatus Desc` and `Order item value`. Define masks:
  $$ \text{cancelled} = (\text{casefold}(\text{strip}(`BillingStatus Desc`)) = \text{"cancelled"}) $$
  $$ \text{negative} = (`Order item value` < 0) $$
- Remove rows where $\text{cancelled} \lor \text{negative}$.

### 3) Consolidating Duplicate Material Lines (same order + ship-to [+ UoM])
- Group keys: $(\text{`Sales Document Number`},\ \text{`Material Number`},\ \text{`Ship-To Party`}[,\ \text{`Sales unit`}])$.
- Within each group, aggregate:
  - Sum quantitative fields: `Sales Order item qty`, `Order item value`, `Invoiced value`, `Actual quantity delivered`, `Quantity invoiced`, `Value Open`, `Subtotal 1..6`, `Confirmed Quantity`.
  - Take the first value (stable time order) for other columns.
  - Recompute unit price when possible:
    $$ \text{`Unit Price`} \leftarrow \begin{cases}
        \dfrac{\sum \text{`Order item value`}}{\sum \text{`Sales Order item qty`}} & \text{if } \sum \text{qty} > 0 \\
        \text{first(`Unit Price`)} & \text{otherwise}
    \end{cases} $$

### 4) First-Time Customer–Material Orders
- For each $(c,m) = (\text{`Sold To number`}, \text{`Material Number`})$, compute the first order date:
  $$ T^{\text{first}}_{c,m} = \min\bigl\{ \text{`Sales Document Created Date`} \bigr\} $$
- Define
  $$ \text{`is\_first\_time\_cust\_material\_order`} = \mathbf{1}\{\ \text{`Sales Document Created Date`} = T^{\text{first}}_{c,m}\ \} $$


### 5) Quantity Deviation Features (per customer–material)
- For each pair $(c,m)$, compute statistics of `Sales Order item qty` (denote $Q$):
  $$\mu_{c,m} = \operatorname{mean}(Q),\quad \sigma_{c,m} = \operatorname{std}(Q),\quad Q^{c,m}_{0.05} = \operatorname{quantile}_{0.05}(Q),\quad Q^{c,m}_{0.95} = \operatorname{quantile}_{0.95}(Q)$$
- For each order with quantity $q$:
  $$ \text{`qty\_deviation\_from\_mean`} = q - \mu_{c,m} $$
  $$ \text{`qty\_z\_score`} = \begin{cases}
       \dfrac{q - \mu_{c,m}}{\sigma_{c,m}} & \text{if } \sigma_{c,m} > 0 \\
       0 & \text{otherwise}
     \end{cases} $$
  $$ \text{`is\_qty\_outside\_typical\_range`} = \mathbf{1}\{\ q < Q^{c,m}_{0.05}\ \lor\ q > Q^{c,m}_{0.95}\ \} $$

### 6) Unusual Unit of Measure (UoM)
- Historical common UoM for each $(c,m)$:
  $$ U^{\text{mode}}_{c,m} = \operatorname{mode}(\text{`Sales unit`}) $$
- Define
  $$ \text{`is\_unusual\_uom`} = \mathbf{1}\{\ \text{`Sales unit`} \ne U^{\text{mode}}_{c,m}\ \} $$

### 7) Suspected Duplicate Orders (24-hour rule)
- Sort by $(c,m,q,t)$ where $q = \text{`Sales Order item qty`}$, $t=\text{`Sales Document Created Date`}$.
- Within groups $(c,m,q)$, define the previous order timestamp and IDs using 1-step lag.
- Compute time difference in hours:
  $$ \Delta t = \frac{t - t_{\text{prev}}}{3600\ \text{seconds/hour}} $$
- Define duplicate flag:
  $$ \text{`is\_suspected\_duplicate\_order`} = \mathbf{1}\{\ 0 < \Delta t \le 24\ \} $$

### 8) Monthly Volume Context (Rolling z-score) and Order Share
- Month index: $\tau = \operatorname{to\_period\_month}(\text{`Sales Document Created Date`})$.
- Monthly total quantity for each $(c,m,\tau)$:
  $$ Y_{c,m,\tau} = \sum_{i \in \text{orders}(c,m,\tau)} q_i $$
- Rolling baseline per $(c,m)$, excluding current month from baseline. Let the trailing window be up to $W=6$ months, and require at least $W_{\min}=3$ prior months for a baseline:
  $$ \mu^{\text{roll}}_{c,m,\tau} = \operatorname{mean}\bigl( Y_{c,m,\tau-1}, Y_{c,m,\tau-2}, \dots \bigr) $$
  $$ \sigma^{\text{roll}}_{c,m,\tau} = \operatorname{std}\bigl( Y_{c,m,\tau-1}, Y_{c,m,\tau-2}, \dots \bigr) $$
  $$ \text{`month\_rolling\_z`} = Z^{\text{roll}}_{c,m,\tau} = \frac{Y_{c,m,\tau} - \mu^{\text{roll}}_{c,m,\tau}}{\sigma^{\text{roll}}_{c,m,\tau}} $$
  If fewer than $W_{\min}$ prior months exist or $\sigma^{\text{roll}}_{c,m,\tau} = 0$, the z-score is undefined (stored as missing).
- Per-order share of month:
  For an order with quantity $q_i$ in $(c,m,\tau)$:
  $$ \text{`order\_share\_of\_month`} = s_i = \begin{cases}
       \dfrac{q_i}{Y_{c,m,\tau}} & \text{if } Y_{c,m,\tau} > 0 \\
       0 & \text{otherwise}
     \end{cases} $$
- Order-level high deviation flag (based on previously computed order z):
  $$ \text{`is\_order\_qty\_high\_z`} = \mathbf{1}\{\ |\text{`qty\_z\_score`}| \ge 2.0\ \} $$

### 9) Unusual Delivery Destination (Ship-To Rarity)
- For each Sold-To $c$ and Ship-To $s$: counts and percentage
  $$ N_{c,s} = \text{count of rows with (c,s)}\ ,\quad N_c = \sum_s N_{c,s} $$
  $$ p_{c,s} = \frac{N_{c,s}}{N_c} $$
- Define
  $$ \text{`ship\_to\_percentage\_for\_sold\_to`} = p_{c,s} \ ,\quad \text{`is\_unusual\_ship\_to\_for\_sold\_to`} = \mathbf{1}\{\ p_{c,s} < 0.01\ \} $$

### 10) Pricing Features
- Unit price distribution per `(Material Number, Sales unit)` with quantiles:
  $$ P^{m,u}_{0.05},\ P^{m,u}_{0.95} $$
  Flag unusual prices:
  $$ \text{`is\_unusual\_unit\_price`} = \mathbf{1}\{\ \text{`Unit Price`} < P^{m,u}_{0.05}\ \lor\ \text{`Unit Price`} > P^{m,u}_{0.95}\ \} $$
- Expected order value and mismatch flag:
  $$ \text{`expected\_order\_item\_value`} = \text{`Unit Price`} \times \text{`Sales Order item qty`} $$
  $$ \text{`is\_value\_mismatch\_price\_qty`} = \mathbf{1}\{\ |\text{`Order item value`} - \text{`expected\_order\_item\_value`}| > 0.01\ \} $$

### 11) Fulfillment Time Features
- Fulfillment duration in days:
  $$ \text{`fulfillment\_duration\_days`} = (\text{`Actual GI Date`} - \text{`Sales Document Created Date`})_{\text{days}} $$
- Per `(Material Number, Ship-To Party)` quantiles:
  $$ F^{m,s}_{0.05},\ F^{m,s}_{0.95} $$
  Flag unusual fulfillment time:
  $$ \text{`is\_unusual\_fulfillment\_time`} = \mathbf{1}\{\ \text{`fulfillment\_duration\_days`} < F^{m,s}_{0.05}\ \lor\ \text{`fulfillment\_duration\_days`} > F^{m,s}_{0.95}\ \} $$

### 12) Human-Readable Anomaly Explanations
For each row, construct a semicolon-separated explanation by appending messages for the following conditions when true:
- `is_first_time_cust_material_order`
- `is_rare_material`
- `is_qty_outside_typical_range` (includes values and $[Q^{c,m}_{0.05}, Q^{c,m}_{0.95}]$)
- `is_unusual_uom`
- `is_suspected_duplicate_order` (includes prior order info and $\Delta t$ in hours)
- Month context: if `month_rolling_z` is defined, append its value
- `is_unusual_ship_to_for_sold_to` (includes percentage)
- `is_unusual_unit_price` (includes $[P^{m,u}_{0.05}, P^{m,u}_{0.95}]$)
- `is_value_mismatch_price_qty` (includes actual vs. expected values)
- `is_unusual_fulfillment_time` (includes $[F^{m,s}_{0.05}, F^{m,s}_{0.95}]$)

### 13) Exported Columns (Selected Subset)
The selected export includes identifiers, timestamps, and the key features above, notably:
- Quantity context: `Sales Order item qty`, `hist_mean`, `hist_std`, `p05`, `p95`, `qty_deviation_from_mean`, `qty_z_score`, `is_qty_outside_typical_range`
- Month context: `current_month_total_qty`, `month_rolling_mean`, `month_rolling_std`, `month_rolling_z`, `order_share_of_month`, `is_order_qty_high_z`
- UoM: `Sales unit`, `historical_common_uom`, `is_unusual_uom`
- Duplicates: `is_suspected_duplicate_order`
- Delivery rarity: `ship_to_percentage_for_sold_to`, `is_unusual_ship_to_for_sold_to`
- Pricing: `Unit Price`, `price_p05`, `price_p95`, `is_unusual_unit_price`, `Order item value`, `expected_order_item_value`, `is_value_mismatch_price_qty`
- Fulfillment: `fulfillment_duration_days`, `fulfillment_p05`, `fulfillment_p95`, `is_unusual_fulfillment_time`
- `anomaly_explanation`


### Reasoning Summary
- Use per-customer–material statistics for order-level deviations to maintain specificity.
- Use a rolling (current-excluded) month baseline to provide contextual, recency-aware month signals without asserting boolean month anomalies for every order.
- Add per-order contribution (`order_share_of_month`) so the model can distinguish small vs. dominant orders within anomalous months.
- Guard rarity and quantile-based checks with sensible thresholds (e.g., 1% ship-to, absolute value mismatch tolerance, and implicit protections via rolling std and min periods).


