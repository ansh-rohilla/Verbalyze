# ASR v3 Domain Data Guide

This document lists domain-specific short forms, phrases, ambiguity cases, and transcript examples for future ASR data collection and synthetic data generation.

## Data Collection Priorities

Collect examples that cover:

```text
short forms and acronyms
spoken vs written forms
numbers, dates, time, and amounts
Indian names and addresses
domain-specific vocabulary
punctuation and casing
question tags
hyphenated words
code-switching
noisy real call conditions
```

## Transcript Style Policy

Before collecting or generating data, decide the canonical written output format.

Examples:

```text
spoken: k y c
written: KYC

spoken: u p i
written: UPI

spoken: rupees five thousand four hundred twenty
written: Rs. 5,420

spoken: mr sharma
written: Mr. Sharma

spoken: maam
written: ma'am
```

Keep the same style across all training, validation, and evaluation transcripts.

## Banking And Financial Short Forms

```text
KYC - Know Your Customer
PAN - Permanent Account Number
Aadhaar
UIDAI
UPI - Unified Payments Interface
IFSC
MICR
NEFT
RTGS
IMPS
ATM
PIN
OTP
CVV
EMI
ECS
NACH
FD - Fixed Deposit
RD - Recurring Deposit
CASA
NPA
ROI - Rate of Interest
APR
LTV
CIBIL
GST
TDS
TAN
TIN
SIP
NAV
AMC
NFO
IPO
KRA
CKYC
PMS
AUM
NRE
NRO
FCNR
POS
QR
BBPS
NPCI
RBI
SEBI
IRDAI
```

## Banking And Financial Phrases

```text
KYC update
PAN verification
Aadhaar seeding
UPI transaction
failed transaction
refund initiated
payment pending
auto debit
mandate registration
loan application
personal loan
home loan
vehicle loan
credit card limit
debit card blocked
ATM withdrawal
minimum balance
account statement
interest rate
processing fee
foreclosure charges
prepayment charges
overdue EMI
late payment fee
CIBIL score
fixed deposit maturity
recurring deposit maturity
nominee update
branch visit
net banking
mobile banking
registered mobile number
beneficiary account
fund transfer
transaction reference number
chargeback request
fraud dispute
account freeze
account closure
```

## Healthcare Short Forms

```text
BP - Blood Pressure
HR - Heart Rate
RR - Respiratory Rate
SpO2
ECG
EKG
EEG
MRI
CT scan
X-ray
CBC
LFT
KFT
RBS
FBS
PPBS
HbA1c
TSH
T3
T4
HDL
LDL
BMI
ICU
NICU
OPD
IPD
ER
ED
ENT
OB-GYN
IV
IM
SOS
BD
OD
TDS
QID
PRN
Rx
Dx
Hx
CNS
COPD
TB
UTI
HIV
COVID-19
```

## Healthcare Phrases

```text
blood pressure reading
sugar level
fasting blood sugar
postprandial sugar
thyroid profile
liver function test
kidney function test
complete blood count
CT scan report
MRI report
chest X-ray
ECG abnormality
follow-up appointment
doctor consultation
patient history
current symptoms
medicine dosage
take one tablet twice daily
admitted to ICU
discharge summary
insurance approval
cashless claim
pre-authorization
lab report pending
medical history
allergic reaction
chief complaint
vital signs
oxygen saturation
```

## Common Honorifics And Education Terms

These are useful across finance, banking, healthcare, and education-related conversations.

```text
Mr.
Mrs.
Ms.
Miss
ma'am
Dr.
Prof.
BTech
MTech
MS
MBA
PhD
```

## Ambiguous Short Forms

Some short forms have different meanings depending on the domain. These need context-aware training examples.

```text
MS
- Master of Science in education
- Multiple sclerosis in healthcare
- Microsoft in business or technology

BP
- Blood pressure in healthcare
- Basis points in finance

TDS
- Tax deducted at source in finance
- Three times daily in healthcare prescriptions

OD
- Once daily in healthcare
- Overdraft in banking

ROI
- Rate of interest in banking
- Return on investment in finance or business

FD
- Fixed deposit in banking
- Final diagnosis in some medical contexts
```

## Spoken To Written Examples

Include both expanded spoken forms and acronym spoken forms.

```text
spoken: k y c
written: KYC

spoken: know your customer
written: KYC

spoken: u p i
written: UPI

spoken: unified payments interface
written: UPI

spoken: b p
written: BP

spoken: blood pressure
written: blood pressure

spoken: one forty over ninety
written: 140/90

spoken: h b a one c
written: HbA1c

spoken: cibil score
written: CIBIL score

spoken: i f s c code
written: IFSC code
```

## Banking Transcript Examples

```text
Your KYC is pending, ma'am.
Please enter the OTP sent to your registered mobile number.
Your EMI of Rs. 5,420 is due on June 20.
Mr. Sharma, your PAN verification has failed.
Did you authorize this UPI transaction?
The IFSC code is HDFC0001234.
Your debit card has been blocked for security reasons.
The refund has been initiated to your source account.
Your CIBIL score is required for this loan application.
Would you like to update the nominee for your FD?
The auto debit mandate is active on your account.
Please confirm whether this transaction was done by you.
```

## Financial Services Transcript Examples

```text
Your SIP installment will be debited tomorrow.
The NAV was updated after market close.
This NFO closes on Friday.
Your portfolio AUM is Rs. 12.5 lakh.
The ROI for this investment is higher than expected.
SEBI guidelines require updated KYC details.
Your GST and TDS details are missing from the form.
The IPO application was submitted through UPI.
Your AMC has sent the account statement by email.
```

## Healthcare Transcript Examples

```text
Your BP is 140/90.
The ECG report looks normal.
Please get a CBC and LFT done tomorrow.
She was admitted to the ICU yesterday.
Take this medicine BD for five days.
Your HbA1c is 7.2%.
The CT scan report is pending.
Please book a follow-up appointment with Dr. Mehta.
The patient has a history of COPD.
His SpO2 dropped to 92%.
The doctor advised an MRI of the spine.
The discharge summary will be ready by evening.
```

## Question Tags And Punctuation Examples

These examples help the model learn where to place question marks, commas, and full stops.

```text
You submitted the KYC form, didn't you?
This UPI transaction was not done by you, right?
Your EMI is due today, isn't it?
Please confirm your registered mobile number, ma'am.
Mr. Sharma, your credit card payment is overdue.
The ECG report is normal, but the BP is high.
You took the medicine after food, didn't you?
The doctor asked for a CBC, LFT, and KFT.
```
