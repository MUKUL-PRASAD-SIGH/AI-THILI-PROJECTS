# Evaluation: Public FAQ Answer Bot

- Base model: `openai-community/gpt2`
- Adapter: r=16, alpha=32.0, targets attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
- FAQs: 60 across 10 departments; decoding is greedy

Metrics are explained in `app/evaluate.py`. *Held-out* questions are paraphrases never seen in training; *seen* questions are the canonical wordings used in training. For **unsupported numbers**, lower is better.

## Held-out paraphrases

| System | Fact recall | All facts | Relevant | Unsupported numbers | Token F1 | ROUGE-L | Exact |
|---|---|---|---|---|---|---|---|
| GPT-2, no fine-tuning | 1.4% | 0.0% | 13.3% | 16.7% | 16.6% | 14.4% | 0.0% |
| Word-overlap lookup | 49.3% | 45.0% | 45.0% | 50.0% | 55.4% | 53.7% | 45.0% |
| LoRA GPT-2 (adapter) | 56.7% | 38.3% | 60.0% | 48.3% | 63.3% | 61.0% | 25.0% |

## Seen questions

| System | Fact recall | All facts | Relevant | Unsupported numbers | Token F1 | ROUGE-L | Exact |
|---|---|---|---|---|---|---|---|
| GPT-2, no fine-tuning | 1.0% | 0.0% | 15.0% | 15.0% | 16.2% | 14.3% | 0.0% |
| Word-overlap lookup | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| LoRA GPT-2 (adapter) | 86.4% | 73.3% | 91.7% | 18.3% | 90.8% | 90.4% | 70.0% |

## Held-out accuracy by department (LoRA model)

| Department | n | Fact recall | All facts | Relevant |
|---|---|---|---|---|
| Building Permits | 7 | 67.9% | 57.1% | 100.0% |
| General | 4 | 66.7% | 25.0% | 50.0% |
| Motor Vehicles | 8 | 58.3% | 25.0% | 50.0% |
| Out of scope | 2 | 16.7% | 0.0% | 0.0% |
| Parks and Recreation | 6 | 61.1% | 16.7% | 83.3% |
| Property Tax | 6 | 88.9% | 83.3% | 100.0% |
| Public Library | 6 | 26.4% | 16.7% | 16.7% |
| Vital Records | 5 | 50.0% | 40.0% | 40.0% |
| Waste and Recycling | 8 | 56.2% | 50.0% | 62.5% |
| Water and Sewer | 8 | 50.0% | 37.5% | 50.0% |

## Sample answers to held-out questions

**Q (Motor Vehicles):** My driver's licence is expiring, how can I renew it?

- *Approved:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32.
- *GPT-2, no fine-tuning:* The driver's licence is expiring. — missing online, 90 days, 12 Elm Street, $32
- *Word-overlap lookup:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32. — all facts
- *LoRA GPT-2 (adapter):* Renew online at riverton.gov/dmv up to 90 days before it expires. A $25 account setup fee is added to your first bill. — missing 12 Elm Street, $32

**Q (Motor Vehicles):** What time does the motor vehicles office open?

- *Approved:* The DMV office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon.
- *GPT-2, no fine-tuning:* The office opens at 7:30 a.m. on the first Monday of every month. — missing Monday to Friday, 8 am, 4:30 pm, Saturday
- *Word-overlap lookup:* The DMV office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon. — all facts
- *LoRA GPT-2 (adapter):* The motor vehicle office opens every day from 8 am to 5 pm. — missing Monday to Friday, 4:30 pm, Saturday

**Q (Water and Sewer):** What could cause an unusually large water bill?

- *Approved:* A high bill is often caused by a leak, such as a running toilet. Check your meter with all taps off; if it still moves, you may have a leak. You can request a free leak check by calling 311.
- *GPT-2, no fine-tuning:* The water bill is a major factor in the water bill for many people. The water bill is a major factor in the water bill for many people. — missing leak, toilet, meter, free leak check, 311
- *Word-overlap lookup:* Water bills are sent every two months and are due 20 days after the bill date. A 5 percent late fee applies after that. — missing leak, toilet, meter, free leak check, 311
- *LoRA GPT-2 (adapter):* A high bill is often caused by a leak, such as a running toilet. Check your meter with all taps off; if it still moves, you may have a leak. You can request a free leak check by calling 311. — all facts

**Q (Water and Sewer):** Are there rules about using sprinklers?

- *Approved:* From May to September, lawn watering is allowed only before 10 am or after 6 pm. Even-numbered addresses water on even dates and odd-numbered addresses on odd dates.
- *GPT-2, no fine-tuning:* No, there are no rules about using sprinklers. — missing May to September, 10 am, 6 pm, even, odd
- *Word-overlap lookup:* Call the 24-hour line at 555-0142. The city clears blockages in the public main for free; blockages in the pipe on your property are the owner's responsibility. — missing May to September, 10 am, 6 pm, even, odd
- *LoRA GPT-2 (adapter):* Use the green sprinkler code at riverton.gov/water at least 3 days before you move out. Use the red one on your first move. — missing May to September, 10 am, 6 pm, even, odd

**Q (Waste and Recycling):** Who do I tell about a missed trash pickup?

- *Approved:* Report a missed collection by calling 311 or at riverton.gov/trash by noon the next working day. A crew will return within 2 working days.
- *GPT-2, no fine-tuning:* The City of Toronto has a trash pickup program. The City of Toronto has a trash pickup program. — missing 311, noon, 2 working days
- *Word-overlap lookup:* Call the 24-hour line at 555-0142. The city clears blockages in the public main for free; blockages in the pipe on your property are the owner's responsibility. — missing 311, noon, 2 working days
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
- *LoRA GPT-2 (adapter):* Yes. Deck permits are required for any deck more than 30 inches above the ground or attached to the house. — all facts

**Q (Building Permits):** How tall can my fence be without a permit?

- *Approved:* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one.
- *GPT-2, no fine-tuning:* The City of Toronto has a permit for a fence that is at least 5 feet high. The City of Toronto has a permit for a fence that is at least 5 feet high. — missing 6 feet, 4 feet, front
- *Word-overlap lookup:* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one. — all facts
- *LoRA GPT-2 (adapter):* Fences up to 6 feet tall in back yards and 4 feet tall in front yards do not need a permit. Taller fences require one. — all facts

**Q (Vital Records):** How fast are vital records processed?

- *Approved:* In-person requests are usually ready the same day. Online and mail orders are processed within 7 business days plus mailing time.
- *GPT-2, no fine-tuning:* The City of Toronto's system of record processing is designed to ensure that records are processed in a timely manner. The City of Toronto's system of record processing is designed to ensure that records are processed in a timely manner. — missing same day, 7 business days
- *Word-overlap lookup:* Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32. — missing same day, 7 business days
- *LoRA GPT-2 (adapter):* Vital records are processed within 5 business days of the date on your request. Once a day, your request is reviewed and revised twice if no one else has reviewed it. — missing same day, 7 business days

**Q (Parks and Recreation):** Do dogs need to be leashed in parks?

- *Approved:* Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill.
- *GPT-2, no fine-tuning:* Yes, dogs need to be leashed in parks. — missing 6 feet, Oak Hill
- *Word-overlap lookup:* Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill. — all facts
- *LoRA GPT-2 (adapter):* Yes. A leash is required to keep a dog off the property. Off-leash play is allowed only in the fenced dog park at Oak Hill. — missing 6 feet

**Q (Public Library):** What happens if my library book is overdue?

- *Approved:* The library does not charge late fees. Items more than 30 days overdue are billed as lost, and the charge is removed when the item is returned.
- *GPT-2, no fine-tuning:* The library will notify you of the overdue book. — missing does not charge, 30 days, lost, returned
- *Word-overlap lookup:* Meeting rooms can be booked online up to 30 days ahead. They are free for non-profit groups and $25 an hour for businesses. — missing does not charge, lost, returned
- *LoRA GPT-2 (adapter):* Book a library book by calling 311 or at riverton.gov/booking, or ask the public for a printed copy at no charge. — missing does not charge, 30 days, lost, returned

**Q (General):** What number do I call for non-emergency city services?

- *Approved:* 311 is Riverton's non-emergency line for city services. Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. For emergencies, call 911.
- *GPT-2, no fine-tuning:* The number of non-emergency city services is determined by the city's emergency management agency. — missing 555-0311, 7 am to 9 pm, 911
- *Word-overlap lookup:* City Hall at 1 Main Street is open Monday to Friday from 8 am to 5 pm, except public holidays. — missing non-emergency, 555-0311, 7 am to 9 pm, 911
- *LoRA GPT-2 (adapter):* 311 is Riverton's non-emergency line for city services. Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. For emergencies, call 911. — all facts

## Where the LoRA model went wrong (held-out)

- **My driver's licence is expiring, how can I renew it?** (missing 12 Elm Street, $32; unsupported $25)
  - got: Renew online at riverton.gov/dmv up to 90 days before it expires. A $25 account setup fee is added to your first bill.
  - approved: Renew online at riverton.gov/dmv up to 90 days before it expires, or visit the DMV office at 12 Elm Street. The renewal fee is $32.
- **What ID is required to apply for a driver's licence?** (missing proof of identity, two proofs, Social Security; unsupported 18, 90)
  - got: Anyone 18 years or older with a valid photo ID can apply for a passport and proof of address at riverton.gov/dmv. Once a passport is valid, the licence is valid for 90 days.
  - approved: Bring one proof of identity such as a passport or birth certificate, two proofs of Riverton address, and your Social Security card.
- **How much will registering my vehicle cost?** (missing $30; unsupported $60)
  - got: Vehicle registration costs $48 per year for a passenger car and $60 per year for a motorcycle.
  - approved: Vehicle registration costs $48 per year for a passenger car and $30 per year for a motorcycle.
- **How do I replace a lost licence plate?** (missing police, $15; unsupported $20, 311)
  - got: Replace a damaged plate with new one, or replace one damaged one online at riverton.gov/park by calling 311. Replacement plates cost $20.
  - approved: Report a stolen plate to the police first, then request replacement plates at the DMV office. Replacement plates cost $15.
- **What time does the motor vehicles office open?** (missing Monday to Friday, 4:30 pm, Saturday; unsupported 5)
  - got: The motor vehicle office opens every day from 8 am to 5 pm.
  - approved: The DMV office is open Monday to Friday from 8 am to 4:30 pm, and on Saturday from 9 am to noon.
- **Can I update my address on my driving licence?** (missing no fee)
  - got: Update your address online at riverton.gov/dmv within 30 days of moving. The DMV office does not update addresses.
  - approved: Update your address online at riverton.gov/dmv within 30 days of moving. There is no fee for an address change.
- **How long do I have to pay my water bill?** (missing two months, 20 days, 5 percent; unsupported 1)
  - got: Water bills are charged 1 percent interest per month on the bill amount, starting the day after the bill date.
  - approved: Water bills are sent every two months and are due 20 days after the bill date. A 5 percent late fee applies after that.
- **Is the city's drinking water safe?** (missing Yes)
  - got: The Riverton drinking water meets all federal and state drinking water standards. The annual water quality report is published every July at riverton.gov/water.
  - approved: Yes. Riverton tap water meets all federal and state drinking water standards. The annual water quality report is published every July at riverton.gov/water.
- **I'm moving away, how do I end my water service?** (missing final meter reading, 3 business days, new address; unsupported 24, 555-0142)
  - got: Call the 24-hour water emergency line at 555-0142. Crews respond to main breaks around the clock.
  - approved: Request a final meter reading online at riverton.gov/water at least 3 business days before you move out. Your final bill is mailed to your new address.
- **Who handles a blocked sewer line?** (missing public main, free, owner)
  - got: Call the 24-hour line at 555-0142. Crews respond to main breaks around the clock.
  - approved: Call the 24-hour line at 555-0142. The city clears blockages in the public main for free; blockages in the pipe on your property are the owner's responsibility.
- **Are there rules about using sprinklers?** (missing May to September, 10 am, 6 pm, even, odd; unsupported 3)
  - got: Use the green sprinkler code at riverton.gov/water at least 3 days before you move out. Use the red one on your first move.
  - approved: From May to September, lawn watering is allowed only before 10 am or after 6 pm. Even-numbered addresses water on even dates and odd-numbered addresses on odd dates.
- **Is there bulky item pickup for a mattress?** (missing 311, 2 free, $35; unsupported 7)
  - got: A bulky item pickup is usually ready the same day. A crew will return within 7 days.
  - approved: Schedule a bulky item pickup by calling 311. Each household gets 2 free bulky pickups per year; extra pickups cost $35.
- **Can I order an extra recycling bin?** (missing 311, free, $60; unsupported $35)
  - got: Bring a bin with photo ID and proof of address to the bin. Each bin costs $35.
  - approved: Request a bin through 311. Replacing a damaged bin is free; an extra trash bin costs $60 and an extra recycling bin is free.
- **When is garden waste picked up?** (missing April to November, green bin, paper)
  - got: The annual collection of yard waste is done each May at Oak Hill Yard. The annual collection is done every other week.
  - approved: Yard waste is collected every other week from April to November. Use the green bin or paper yard bags; plastic bags are not accepted.
- **Will my bin be collected on a holiday?** (missing one day later; unsupported 30)
  - got: Yes. A crew collects household items on a holiday, such as a baby or toddler, and returns the item within 30 days.
  - approved: When a holiday falls on a weekday, collection for the rest of that week moves one day later.
- **Can I dispute my property valuation?** (missing 45 days, similar homes; unsupported 30)
  - got: Property valuation is reviewed every two years and final offers are sent to the Board of Assessment Review within 30 days of the date on the final appraisal notice.
  - approved: File an appeal with the Board of Assessment Review within 45 days of the date on your assessment notice. Include recent sales of similar homes as evidence.
- **Can I replace my roof shingles without a permit?** (missing 25 percent; unsupported 30, 6)
  - got: Repairs smaller than 6 inches do not need a permit. A permit is required to replace more than 30 percent of a roof.
  - approved: A permit is required to replace more than 25 percent of a roof. Repairs smaller than that do not need a permit.
- **When can an inspector come out?** (missing permit portal, 555-0177, 3 pm)
  - got: An inspector is responsible for investigating hazardous waste, chemicals and human health. Inspections are done the next working day.
  - approved: Request an inspection in the permit portal or by calling 555-0177 by 3 pm. Inspections are done the next working day.
- **How long do I have to finish work after getting a permit?** (missing 180 days, 2 years)
  - got: Once a permit is valid, work starts the next working day. Once work starts, the permit is valid for 2 working days.
  - approved: A permit expires if work does not start within 180 days. Once work starts, the permit is valid for 2 years.
- **How can I order copies of a death certificate?** (missing Immediate family, $5)
  - got: Order a certified copy online at riverton.gov/records or in person at the Vital Records office in City Hall. Each copy costs $20.
  - approved: Immediate family members or the estate's representative can order death certificates from Vital Records. The first copy costs $20 and each extra copy ordered at the same time costs $5.
- **How fast are vital records processed?** (missing same day, 7 business days; unsupported 5)
  - got: Vital records are processed within 5 business days of the date on your request. Once a day, your request is reviewed and revised twice if no one else has reviewed it.
  - approved: In-person requests are usually ready the same day. Online and mail orders are processed within 7 business days plus mailing time.
- **Can an error on a certificate be amended?** (missing amendment form, documents, $25; unsupported 7)
  - got: An amendment to a document is required to reopen the certificate. Repairs are usually made within 7 days.
  - approved: Submit an amendment form to Vital Records with documents that prove the correct information. The amendment fee is $25.
- **How do I rent a park shelter?** (missing $50; unsupported $48)
  - got: Reserve shelters online at riverton.gov/parks up to 6 months ahead. Fees are $48 for a half day and $90 for a full day.
  - approved: Reserve shelters online at riverton.gov/parks up to 6 months ahead. Fees are $50 for a half day and $90 for a full day.
- **When can I go swimming at the city pool?** (missing 11 am to 7 pm; unsupported 8)
  - got: The city pool is open from Memorial Day to Labor Day, daily from 11 am to 8 pm. Entry is $4 for adults and $2 for children.
  - approved: The Riverside Pool is open from Memorial Day to Labor Day, daily from 11 am to 7 pm. Entry is $4 for adults and $2 for children.
- **How can I enrol my kids in the city day camp?** (missing March 1, online, fee assistance; unsupported 2, 5, 6, 8)
  - got: Bring one child under 6 months old to City Hall, open Saturdays from 8 am to 5 pm. Bring one adult and 2 children.
  - approved: Summer camp registration opens online on March 1 at riverton.gov/parks. Need-based fee assistance is available.
- **Do dogs need to be leashed in parks?** (missing 6 feet)
  - got: Yes. A leash is required to keep a dog off the property. Off-leash play is allowed only in the fenced dog park at Oak Hill.
  - approved: Dogs are welcome in city parks on a leash no longer than 6 feet. Off-leash play is allowed only in the fenced dog park at Oak Hill.
- **Can I be in a city park at night?** (missing 10 pm; unsupported 9)
  - got: Yes. City parks are open from 6 am to 9 pm every day.
  - approved: City parks are open from 6 am to 10 pm every day.
- **What do I need to join the library?** (missing photo ID, proof of address; unsupported $25, 90)
  - got: Library members can apply online at riverton.gov/parks up to 90 days before they move in. They are free to use for non-profit groups and $25 an hour for businesses.
  - approved: Library cards are free for Riverton residents. Bring a photo ID and proof of address to any branch.
- **How many weeks can I keep a library book?** (missing 3 weeks, renewed twice, 1 week; unsupported $25)
  - got: Library books are free for non-profit groups and $25 an hour for businesses.
  - approved: Books can be borrowed for 3 weeks and renewed twice if no one else has placed a hold. DVDs are loaned for 1 week.
- **What happens if my library book is overdue?** (missing does not charge, 30 days, lost, returned; unsupported 311)
  - got: Book a library book by calling 311 or at riverton.gov/booking, or ask the public for a printed copy at no charge.
  - approved: The library does not charge late fees. Items more than 30 days overdue are billed as lost, and the charge is removed when the item is returned.
- **What time does the main library close?** (missing 9 am to 8 pm, 9 am to 5 pm, 1 pm to 5 pm; unsupported 12)
  - got: The Main Library at 12 Elm Street is open Monday to Friday from 8 am to 5 pm, except public holidays.
  - approved: The Main Library is open Monday to Thursday from 9 am to 8 pm, Friday and Saturday from 9 am to 5 pm, and Sunday from 1 pm to 5 pm.
- **Is there free internet at the library?** (missing 2 hours, Wi-Fi, 10 cents; unsupported 5, 8, 9)
  - got: Yes. The Riverton Public Library is open Monday to Friday from 8 am to 5 pm, and on Saturday from 9 am to noon.
  - approved: Yes. Any cardholder can use a public computer for 2 hours a day, and free Wi-Fi is available to everyone. Printing costs 10 cents a page.
- **What time does City Hall close?** (missing Monday to Friday, 8 am to 5 pm; unsupported 6)
  - got: City Hall at 1 Main Street is open from 6 am to 8 pm every day.
  - approved: City Hall at 1 Main Street is open Monday to Friday from 8 am to 5 pm, except public holidays.
- **Where can I report damaged roads?** (missing 5 working days; unsupported 24)
  - got: Report damaged roads through 311 or the Riverton 311 app. Unsafe roads are closed off within 24 hours.
  - approved: Report potholes through 311 or the Riverton 311 app with the nearest address. Most potholes are filled within 5 working days.
- **Who do I contact about a street lamp that doesn't work?** (missing pole number; unsupported 555-0311, 9)
  - got: Call 311 from any phone in the city, or 555-0311 from outside, any day from 7 am to 9 pm. Crews respond to calls within 7 days.
  - approved: Report the pole number printed on the streetlight to 311. Repairs are usually made within 7 days.
- **What is a good restaurant near me?** (missing city services, 311; unsupported $25)
  - got: A restaurant is free for non-profit groups and $25 an hour for businesses.
  - approved: I can only answer questions about Riverton city services. For other questions, please call 311 and we will try to point you in the right direction.
- **Can you tell me how to fill in my federal tax return?** (missing legal, professional; unsupported 1)
  - got: Download a Form Tender by calling 311. The deadline is 1 business day.
  - approved: I can't give legal or financial advice. Please contact a licensed professional or call 311 for a list of free community legal services.
