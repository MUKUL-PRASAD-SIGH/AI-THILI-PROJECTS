# Evaluation: Public FAQ Answer Bot

- Base model: `openai-community/gpt2`
- Adapter: r=16, alpha=32.0, targets attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
- FAQs: 60 across 10 departments; decoding is greedy

Metrics are explained in `app/evaluate.py`. *Held-out* questions are paraphrases never seen in training; *seen* questions are the canonical wordings used in training. For **unsupported numbers**, lower is better.

## Held-out paraphrases

| System | Fact recall | All facts | Relevant | Unsupported numbers | Token F1 | ROUGE-L | Exact |
|---|---|---|---|---|---|---|---|
| GPT-2, no fine-tuning | 1.4% | 0.0% | 13.3% | 16.7% | 16.6% | 14.4% | 0.0% |
| Word-overlap lookup | 58.8% | 55.0% | 55.0% | 41.7% | 63.2% | 62.2% | 55.0% |
| LoRA GPT-2 (adapter) | 84.6% | 76.7% | 83.3% | 15.0% | 85.3% | 84.6% | 71.7% |

## Seen questions

| System | Fact recall | All facts | Relevant | Unsupported numbers | Token F1 | ROUGE-L | Exact |
|---|---|---|---|---|---|---|---|
| GPT-2, no fine-tuning | 1.0% | 0.0% | 15.0% | 15.0% | 16.2% | 14.3% | 0.0% |
| Word-overlap lookup | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| LoRA GPT-2 (adapter) | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## Held-out accuracy by department (LoRA model)

| Department | n | Fact recall | All facts | Relevant |
|---|---|---|---|---|
| Building Permits | 7 | 82.1% | 71.4% | 100.0% |
| General | 4 | 91.7% | 75.0% | 75.0% |
| Motor Vehicles | 8 | 95.8% | 87.5% | 100.0% |
| Out of scope | 2 | 0.0% | 0.0% | 0.0% |
| Parks and Recreation | 6 | 91.7% | 83.3% | 83.3% |
| Property Tax | 6 | 83.3% | 83.3% | 83.3% |
| Public Library | 6 | 72.2% | 66.7% | 66.7% |
| Vital Records | 5 | 76.7% | 60.0% | 80.0% |
| Waste and Recycling | 8 | 95.8% | 87.5% | 87.5% |
| Water and Sewer | 8 | 91.7% | 87.5% | 87.5% |

## Sample answers to held-out questions

**Q (Motor Vehicles):** My driver's licence is expiring, how can I renew it?

- *Approved:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32.
- *GPT-2, no fine-tuning:* The driver's licence is expiring. — missing online, 90 days, 12 Elm Street, $32
- *Word-overlap lookup:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32. — all facts
- *LoRA GPT-2 (adapter):* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32. — all facts

**Q (Motor Vehicles):** What time does the motor vehicles office open?

- *Approved:* The DMV office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon.
- *GPT-2, no fine-tuning:* The office opens at 7:30 a.m. on the first Monday of every month. — missing Monday to Friday, 8 am, 4:30 pm, Saturday
- *Word-overlap lookup:* The DMV office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon. — all facts
- *LoRA GPT-2 (adapter):* The Motor Vehicle Office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon. — all facts

**Q (Water and Sewer):** What could cause an unusually large water bill?

- *Approved:* A high bill is often caused by a leak, such as a running toilet. Check your meter with all taps off; if it still moves, you may have a leak. You can request a free leak check by calling 311.
- *GPT-2, no fine-tuning:* The water bill is a major factor in the water bill for many people. The water bill is a major factor in the water bill for many people. — missing leak, toilet, meter, free leak check, 311
- *Word-overlap lookup:* Water bills are sent every two months and are due 20 days after the bill date. A 5 percent late fee applies after that. — missing leak, toilet, meter, free leak check, 311
- *LoRA GPT-2 (adapter):* A high bill is often caused by a leak, such as a running toilet. Check your meter with all taps off; if it still moves, you may have a leak. You can request a free leak check by calling 311. — all facts

**Q (Water and Sewer):** Are there rules about using sprinklers?

- *Approved:* From May to September, lawn watering is allowed only before 10 am or after 6 pm. Even-numbered addresses water on even dates and odd-numbered addresses on odd dates.
- *GPT-2, no fine-tuning:* No, there are no rules about using sprinklers. — missing May to September, 10 am, 6 pm, even, odd
- *Word-overlap lookup:* Call the 24-hour line at 555-0142. The city clears blockages in the public main for free; blockages in the pipe on your property are the owner's responsibility. — missing May to September, 10 am, 6 pm, even, odd
- *LoRA GPT-2 (adapter):* From May to September, lawn watering is allowed only before 10 am or after 6 pm. Even-numbered addresses water on even dates and odd-numbered addresses on odd dates. — all facts

**Q (Waste and Recycling):** Who do I tell about a missed trash pickup?

- *Approved:* Report a missed collection by calling 311 or at riverton.gov/trash by noon the next working day. A crew will return within 2 working days.
- *GPT-2, no fine-tuning:* The City of Toronto has a trash pickup program. The City of Toronto has a trash pickup program. — missing 311, noon, 2 working days
- *Word-overlap lookup:* Report potholes through 311 or the Riverton 311 app with the nearest address. Most potholes are filled within 5 working days. — missing noon, 2 working days
- *LoRA GPT-2 (adapter):* Report a missed collection by calling 311 or at riverton.gov/trash by noon the next working day. A crew will return within 2 working days. — all facts

**Q (Property Tax):** Where do I pay property taxes?

- *Approved:* Pay online at riverton.gov/tax, by mail with the payment slip, or in person at the Treasurer's Office in City Hall. Card payments have a 2.5 percent processing fee.
- *GPT-2, no fine-tuning:* The City of Toronto collects property taxes from the City of Toronto. The City of Toronto collects property taxes from the City of Toronto. — missing online, mail, Treasurer's Office, 2.5 percent
- *Word-overlap lookup:* Property taxes are paid in two instalments, due on March 31 and September 30. — missing online, mail, Treasurer's Office, 2.5 percent
- *LoRA GPT-2 (adapter):* Pay online at riverton.gov/tax, by mail with the payment slip, or in person at the Treasurer's Office in City Hall. Card payments have a 2.5 percent processing fee. — all facts

**Q (Building Permits):** Can I build a deck without a permit?

- *Approved:* Yes, a permit is required for any deck more than 30 inches above the ground or attached to the house.
- *GPT-2, no fine-tuning:* No, you can't build a deck without a permit. — missing Yes, 30 inches, attached
- *Word-overlap lookup:* Yes, a permit is required for any deck more than 30 inches above the ground or attached to the house. — all facts
- *LoRA GPT-2 (adapter):* Yes, a permit is required for any deck more than 30 inches above the ground or attached to the house. — all facts

**Q (Building Permits):** How tall can my fence be without a permit?

- *Approved:* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one.
- *GPT-2, no fine-tuning:* The City of Toronto has a permit for a fence that is at least 5 feet high. The City of Toronto has a permit for a fence that is at least 5 feet high. — missing 6 feet, 4 feet, front
- *Word-overlap lookup:* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one. — all facts
- *LoRA GPT-2 (adapter):* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one. — all facts

**Q (Vital Records):** How fast are vital records processed?

- *Approved:* In-person requests are usually ready the same day. Online and mail orders are processed within 7 business days plus mailing time.
- *GPT-2, no fine-tuning:* The City of Toronto's system of record processing is designed to ensure that records are processed in a timely manner. The City of Toronto's system of record processing is designed to ensure that records are processed in a timely manner. — missing same day, 7 business days
- *Word-overlap lookup:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32. — missing same day, 7 business days
- *LoRA GPT-2 (adapter):* In-person requests are usually ready the same day. Online and mail orders are processed within 7 business days plus mailing time. — all facts

**Q (Parks and Recreation):** Do dogs need to be leashed in parks?

- *Approved:* Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill.
- *GPT-2, no fine-tuning:* Yes, dogs need to be leashed in parks. — missing 6 feet, Oak Hill
- *Word-overlap lookup:* Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill. — all facts
- *LoRA GPT-2 (adapter):* Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill. — all facts

**Q (Public Library):** What happens if my library book is overdue?

- *Approved:* The library does not charge late fees. Items more than 30 days overdue are billed as lost, and the charge is removed when the item is returned.
- *GPT-2, no fine-tuning:* The library will notify you of the overdue book. — missing does not charge, 30 days, lost, returned
- *Word-overlap lookup:* Meeting rooms can be booked online up to 30 days ahead. They are free for non-profit groups and $25 an hour for businesses. — missing does not charge, lost, returned
- *LoRA GPT-2 (adapter):* Request a collection day by calling 311. Each collection day costs $20 and an extra fee is added to your collection. — missing does not charge, 30 days, lost, returned

**Q (General):** What number do I call for non-emergency city services?

- *Approved:* 311 is Riverton's non-emergency line for city services. Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. For emergencies, call 911.
- *GPT-2, no fine-tuning:* The number of non-emergency city services is determined by the city's emergency management agency. — missing 555-0311, 7 am to 9 pm, 911
- *Word-overlap lookup:* 311 is Riverton's non-emergency line for city services. Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. For emergencies, call 911. — all facts
- *LoRA GPT-2 (adapter):* 311 is Riverton's non-emergency line for city services. Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. For emergencies, call 911. — all facts

## Where the LoRA model went wrong (held-out)

- **How do I replace a lost licence plate?** (missing police)
  - got: Request a stolen plate online at riverton.gov/dmv or in person at the DMV office in City Hall. Replacement plates cost $15.
  - approved: Report a stolen plate to the police first, then request replacement plates at the DMV office. Replacement plates cost $15.
- **I'm moving away, how do I end my water service?** (missing final meter reading, new address; unsupported $25)
  - got: Apply online at riverton.gov/water at least 3 business days before you move out. A $25 account setup fee is added to your first bill.
  - approved: Request a final meter reading online at riverton.gov/water at least 3 business days before you move out. Your final bill is mailed to your new address.
- **Can I order an extra recycling bin?** (missing $60; unsupported $35, 2)
  - got: Request a bin through 311. Each household gets 2 free bulky pickups per year; extra pickups cost $35.
  - approved: Request a bin through 311. Replacing a damaged bin is free; an extra trash bin costs $60 and an extra recycling bin is free.
- **Can I dispute my property valuation?** (missing Board of Assessment Review, 45 days, similar homes; unsupported 10)
  - got: Property valuations are reviewed every two months and final destination sales are reviewed within 10 business days.
  - approved: File an appeal with the Board of Assessment Review within 45 days of the date on your assessment notice. Include recent sales of similar homes as evidence.
- **When can an inspector come out?** (missing permit portal, 555-0177, 3 pm)
  - got: Inspections are done the next working day.
  - approved: Request an inspection in the permit portal or by calling 555-0177 by 3 pm. Inspections are done the next working day.
- **How long do I have to finish work after getting a permit?** (missing 180 days)
  - got: Once work starts, the permit is valid for 2 years.
  - approved: A permit expires if work does not start within 180 days. Once work starts, the permit is valid for 2 years.
- **How can I order copies of a death certificate?** (missing Immediate family, $5)
  - got: Order a certified copy online at riverton.gov/records or in person at the Vital Records office in City Hall. Each copy costs $20.
  - approved: Immediate family members or the estate's representative can order death certificates from Vital Records. The first copy costs $20 and each extra copy ordered at the same time costs $5.
- **Can an error on a certificate be amended?** (missing amendment form, documents; unsupported 2)
  - got: An amendment is required to a certificate that is valid for 2 years. The amendment fee is $25.
  - approved: Submit an amendment form to Vital Records with documents that prove the correct information. The amendment fee is $25.
- **Where do I report unsafe play equipment?** (missing 24 hours; unsupported 5)
  - got: Report unsafe play through 311 or the Riverton 311 app with the nearest address. Most unsafe plays are made within 5 working days.
  - approved: Report damaged park equipment through 311 or the Riverton 311 app. Unsafe equipment is closed off within 24 hours.
- **What do I need to join the library?** (missing photo ID, proof of address; unsupported $25, 30)
  - got: Meeting rooms can be booked online up to 30 days ahead. They are free for non-profit groups and $25 an hour for businesses.
  - approved: Library cards are free for Riverton residents. Bring a photo ID and proof of address to any branch.
- **What happens if my library book is overdue?** (missing does not charge, 30 days, lost, returned; unsupported $20, 311)
  - got: Request a collection day by calling 311. Each collection day costs $20 and an extra fee is added to your collection.
  - approved: The library does not charge late fees. Items more than 30 days overdue are billed as lost, and the charge is removed when the item is returned.
- **Where can I report damaged roads?** (missing 5 working days; unsupported 24)
  - got: Report damaged roads through 311 or the Riverton 311 app. Unsafe roads are closed off within 24 hours.
  - approved: Report potholes through 311 or the Riverton 311 app with the nearest address. Most potholes are filled within 5 working days.
- **What is a good restaurant near me?** (missing city services, 311; unsupported 5, 8)
  - got: The Riverside Inn is open Monday to Friday from 8 am to 5 pm, except public holidays.
  - approved: I can only answer questions about Riverton city services. For other questions, please call 311 and we will try to point you in the right direction.
- **Can you tell me how to fill in my federal tax return?** (missing legal, professional, 311)
  - got: Download a copy at riverton.gov/tax using your parcel number, or ask the Treasurer's Office for a printed copy at no charge.
  - approved: I can't give legal or financial advice. Please contact a licensed professional or call 311 for a list of free community legal services.
