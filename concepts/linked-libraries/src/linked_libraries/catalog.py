"""The hand-written library the datasets are drawn from.

Fernhill, a company of about 80 people that sells logistics planning software,
has an assistant that has learned about 195 procedures, kept in 30 skill
folders, and 28 rules about how work is done there. Each rule says what it
applies to as a list of kinds of work, and the procedures it governs (its
links) are exactly the procedures of those kinds. Most rules govern procedures
in several folders.

Each procedure has a docstring that names its kind of work, a task that needs
it (phrased as a person would ask, without naming the procedure), and whether
its folder's description leaves it out ("hidden"). Each rule has one change:
the fact that changes (old and new, for the judge), a direct request, and a
phrase to drop into a task in passing.

generate.py turns this into datasets of several sizes.
"""

from __future__ import annotations

FOLDERS: dict[str, dict[str, str]] = {}
PROCEDURES: list[dict] = []
RULES: dict[str, dict] = {}


def folder(name: str, title: str, description: str) -> None:
    FOLDERS[name] = {"title": title, "description": description}


def proc(folder_name: str, name: str, params: str, returns: str, doc: str, task: str,
         rules: tuple[str, ...] = (), hidden: bool = False) -> None:
    PROCEDURES.append({
        "folder": folder_name, "name": name, "params": params, "returns": returns,
        "doc": doc, "task": task, "rules": list(rules), "hidden": hidden,
    })


def rule(rid: str, title: str, applies: str, text: str, *, old: str, new: str,
         direct: str, passing: str) -> None:
    RULES[rid] = {"id": rid, "title": title, "applies": applies, "text": text,
                  "change": {"old": old, "new": new, "direct": direct, "passing": passing}}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule("R01", "Writing to investors",
     "updates to our investors, investor decks, posts on the investor portal and answers to investors' questions",
     "Lead with cash runway in months, then the number that changed most since the last update, and keep it to one page.",
     old="lead with cash runway",
     new="lead with net revenue retention (instead of cash runway)",
     direct="Change our rule on writing to investors: from now on, lead with net revenue retention instead of cash runway. The rest of the rule stays as it is.",
     passing="Also, from now on anything we write to investors should lead with net revenue retention rather than cash runway, so please make sure that sticks.")

rule("R02", "Naming customers in public material",
     "case studies, blog posts, social media posts and replies, press releases, newsletters, sales decks and customer logo slides",
     "Refer to a customer by its company name, never by an employee's name, and never include anyone's phone number or home address.",
     old="refer to a customer by its company name",
     new="do not name a customer unless they have agreed in writing; describe them by industry instead, for example 'a logistics customer'",
     direct="Tighten our rule on naming customers in public material: we no longer name a customer unless they have agreed in writing. Describe them by their industry instead, for example 'a logistics customer'. The rest of the rule stays.",
     passing="One more thing: from now on we don't name a customer in anything public unless they've agreed in writing, we describe them by industry instead, like 'a logistics customer'. Please make that stick for the future.")

rule("R03", "Meetings with people outside the company",
     "scheduling or moving any meeting with people outside the company, such as customer calls and business reviews, sales demos, job interviews, investor calls, media interviews and supplier or partner meetings",
     "Never schedule them on a Friday, and always put an agenda in the invitation.",
     old="never on a Friday",
     new="never on a Monday (Fridays are now allowed)",
     direct="Change our rule on meetings with people outside the company: they are now off limits on Mondays instead of Fridays. Fridays are fine again. Keep the agenda part.",
     passing="By the way, meetings with people outside the company are now off limits on Mondays rather than Fridays; Fridays are fine again. Please keep that going forward.")

rule("R04", "Stating amounts of money",
     "invoices and payment reminders, quotes and order forms, refunds and credit notes, pricing and sales decks, purchase orders, expense claims and spending reports, payslips and payroll cost reports, budgets and forecasts, revenue reports and profit and loss statements",
     "State every amount in US dollars with two decimals; if a figure was in another currency, convert it at the day's rate and name the rate.",
     old="state amounts in US dollars",
     new="state amounts in euros (not US dollars)",
     direct="Update our rule on stating amounts of money: we now state every amount in euros, not US dollars, still with two decimals and naming the rate when we convert.",
     passing="Oh, and from now on we state every amount in euros rather than US dollars, still with two decimals and naming the rate when we convert. Please make sure that sticks for the future.")

rule("R05", "Alt text for images",
     "images in blog posts, newsletters, event pages and the pages of our website",
     "Every image needs alt text that describes it in under 125 characters.",
     old="alt text under 125 characters",
     new="alt text under 80 characters",
     direct="Change our alt text rule: alt text must now be under 80 characters instead of 125.",
     passing="While you're at it: alt text now has to be under 80 characters, not 125. Please keep to that from now on.")

rule("R06", "Our product's name",
     "copy we publish that names our product: website pages, blog posts, case studies, release notes, newsletters and marketing emails, press releases, job ads and sales decks",
     "Write the product's name as 'Atlas', never 'ATLAS' or 'the Atlas platform'.",
     old="the product is called 'Atlas'",
     new="the product is called 'Atlas Cloud' (never just 'Atlas')",
     direct="Our product has been renamed: update the rule on its name so it is written 'Atlas Cloud', never just 'Atlas', 'ATLAS' or 'the Atlas platform'.",
     passing="Heads up, the product has been renamed: from now on it's 'Atlas Cloud', never just 'Atlas'. Please make sure that's how we write it from here on.")

rule("R07", "Talking to the press",
     "press releases, replies to journalists, story pitches, media interviews and the media list",
     "Everything for the press goes through Maya Chen, our communications lead, who approves it before it goes out.",
     old="Maya Chen approves everything for the press",
     new="Tom Okafor approves everything for the press (he replaced Maya Chen as communications lead)",
     direct="Maya Chen has left. Update our rule on talking to the press: Tom Okafor is now our communications lead and approves everything for the press.",
     passing="Also, Maya has left and Tom Okafor is our communications lead now, so everything for the press goes through him. Please keep that for the future.")

rule("R08", "Contracts",
     "NDAs, customer contracts and order forms, supplier and partner agreements, and contract renewals",
     "Legal (Priya Nair) must approve every contract before it is sent or signed, and it must use the current template from the legal drive.",
     old="legal alone approves contracts",
     new="legal and the CFO must both approve contracts",
     direct="Change our contracts rule: from now on legal and the CFO must both approve a contract before it is sent or signed. The template part stays.",
     passing="One thing for the future: contracts now need sign-off from both legal and the CFO before they go out or get signed. Please make sure that sticks.")

rule("R09", "Refunds and credits",
     "refunds, credit notes and cancellations with a prorated refund",
     "A refund or credit worth more than one month of the customer's subscription needs a finance manager's approval before it is issued.",
     old="approval needed above one month of the subscription",
     new="approval needed above one week of the subscription",
     direct="Tighten our refunds rule: a refund or credit worth more than one week of the customer's subscription, not one month, now needs a finance manager's approval.",
     passing="And from now on any refund or credit worth more than a week of the customer's subscription, not a month, needs a finance manager's approval. Please keep that going forward.")

rule("R10", "Paying suppliers",
     "payments to suppliers and contractors, wire transfers, partner commissions and setting up a new supplier for payment",
     "Every payment needs two approvers from finance, and a new supplier's bank details must be confirmed by phone before the first payment.",
     old="confirm a new supplier's bank details by phone",
     new="confirm a new supplier's bank details through the supplier portal, never by phone",
     direct="Change our rule on paying suppliers: a new supplier's bank details must now be confirmed through the supplier portal, never by phone. The two approvers stay.",
     passing="Also, after the fraud scare, new suppliers' bank details now get confirmed through the supplier portal, never by phone. Please make sure that sticks.")

rule("R11", "Writing to candidates",
     "interview invitations, rejections, offers and follow-ups to job candidates",
     "Reply to every candidate within five working days, and never ask about salary history.",
     old="reply within five working days",
     new="reply within three working days",
     direct="Change our rule on writing to candidates: we now reply to every candidate within three working days instead of five.",
     passing="By the way, we've promised candidates a reply within three working days now, not five. Please keep to that from now on.")

rule("R12", "Exporting personal data",
     "exports of personal data: mailing lists, contact lists, a customer's user records, employee lists, candidate pipelines, ticket requesters and event attendee lists",
     "Record a ticket number for the export, send the file only encrypted, and delete local copies after 30 days.",
     old="delete local copies after 30 days",
     new="delete local copies after 7 days",
     direct="Change our rule on exporting personal data: local copies must now be deleted after 7 days instead of 30.",
     passing="Oh, and local copies of personal data exports now have to be deleted after 7 days, not 30. Please make that stick.")

rule("R13", "Employee records",
     "changes to an employee's HR record: salary, tax details, pay corrections, leave, performance reviews and leavers",
     "Only people in HR may see them, and every file must be marked 'HR confidential'.",
     old="files are marked 'HR confidential'",
     new="files are marked 'Restricted – HR only'",
     direct="Change our rule on employee records: files must now be marked 'Restricted – HR only' instead of 'HR confidential'.",
     passing="Also, HR files are now marked 'Restricted – HR only' instead of 'HR confidential'. Please keep that from now on.")

rule("R14", "Booking travel",
     "booking flights, trains, hotels and hire cars for anyone travelling for the company, including speakers",
     "Book economy class for flights under six hours, and keep hotels under the city's nightly cap in the travel policy.",
     old="economy for flights under six hours",
     new="economy for flights under eight hours",
     direct="Change our travel rule: book economy for flights under eight hours instead of six.",
     passing="And from now on it's economy for any flight under eight hours, not six. Please make sure that sticks.")

rule("R15", "Changing production",
     "deploys, database migrations, feature-flag and configuration changes in production, and rollbacks",
     "Make production changes only between 09:00 and 16:00 UK time on working days, and announce each one in the #deploys channel before it starts.",
     old="announce in #deploys",
     new="announce in #releases (the #deploys channel is retired)",
     direct="Change our rule on production changes: announce each one in #releases from now on; #deploys has been retired.",
     passing="Also, we've retired #deploys, so production changes get announced in #releases from now on. Please keep that going forward.")

rule("R16", "Granting access",
     "granting, changing or revoking access to systems, accounts, shared drives, partner portals and the office building",
     "Grant the least access the person needs, and set it to expire after 90 days unless it is renewed.",
     old="access expires after 90 days",
     new="access expires after 60 days",
     direct="Change our access rule: access now expires after 60 days unless renewed, instead of 90.",
     passing="By the way, access now expires after 60 days unless it's renewed, not 90. Please make that stick.")

rule("R17", "Replying to customers about problems",
     "replies to support tickets and to customer complaints",
     "Reply within four working hours, and sign as 'The Fernhill Support Team'.",
     old="sign as 'The Fernhill Support Team'",
     new="sign with the first name of the person who writes the reply",
     direct="Change our rule on replying to customers about problems: replies are now signed with the first name of the person who writes them, instead of 'The Fernhill Support Team'.",
     passing="Also, from now on we sign these with our own first names, not 'The Fernhill Support Team'. Please keep that going forward.")

rule("R18", "Stating times to people outside the company",
     "invitations to meetings with people outside the company, outage notices, planned maintenance notices and webinar announcements",
     "State every time in UTC and in the recipient's time zone.",
     old="state times in UTC and the recipient's time zone",
     new="state times only in the recipient's time zone, naming their city",
     direct="Change our rule on stating times to people outside the company: give times only in the recipient's time zone, naming their city, and drop UTC.",
     passing="And from now on, when we give people outside the company a time, use only their own time zone and name their city; drop the UTC. Please make that stick.")

rule("R19", "Board materials",
     "board decks, board minutes, board pre-reads and written resolutions",
     "Mark every page 'Confidential – Board', and share it only through the board portal, never as an email attachment.",
     old="pages are marked 'Confidential – Board'",
     new="pages are marked 'Strictly confidential'",
     direct="Change our rule on board materials: mark every page 'Strictly confidential' instead of 'Confidential – Board'. The board portal part stays.",
     passing="Also, board papers are now marked 'Strictly confidential' instead of 'Confidential – Board'. Please keep that for the future.")

rule("R20", "Forecasts",
     "revenue forecasts, cash-flow projections, budgets and pipeline forecasts",
     "Give every forecast as a range with its assumptions stated, never as a single number.",
     old="give forecasts as a range",
     new="give forecasts as three scenarios: low, base and high",
     direct="Change our forecasts rule: give every forecast as three scenarios, low, base and high, instead of a range, still with the assumptions stated.",
     passing="One more thing: forecasts should now come as three scenarios, low, base and high, rather than a range. Please make sure that sticks.")

rule("R21", "Sharing data with other companies",
     "sending datasets, logs or other data to partners, suppliers, auditors or integration providers",
     "Only share data under a signed data processing agreement, remove names and email addresses first, and send it through Vaultline, never as an email attachment.",
     old="send data through Vaultline",
     new="send data through the company SFTP server (no longer Vaultline)",
     direct="Change our rule on sharing data with other companies: we've dropped Vaultline, so send data through the company SFTP server instead.",
     passing="Also, we've dropped Vaultline: data for other companies goes through the company SFTP server now. Please keep that from now on.")

rule("R22", "Buying things",
     "buying things: purchase orders, new software subscriptions and renewals, office supplies, and equipment such as laptops",
     "Every purchase needs a purchase order approved by finance before ordering, and new software must be checked by IT first.",
     old="purchase orders approved by finance",
     new="purchase orders approved by the budget owner",
     direct="Change our purchasing rule: purchase orders are now approved by the budget owner instead of finance. The IT check stays.",
     passing="By the way, purchase orders get approved by the budget owner now, not finance. Please make sure that sticks.")

rule("R23", "Posting on social media",
     "posts, scheduled posts and replies on the company's social accounts",
     "Never post at weekends, and end every post with #BuildWithFernhill.",
     old="end posts with #BuildWithFernhill",
     new="end posts with #FernhillBuilds",
     direct="Change our social media rule: end every post with #FernhillBuilds instead of #BuildWithFernhill.",
     passing="Oh, and our hashtag is now #FernhillBuilds instead of #BuildWithFernhill. Please use that from now on.")

rule("R24", "Shipping goods",
     "shipping laptops and equipment to staff, scanner kits and other hardware to customers, and event materials",
     "Ship international parcels with DHL Express, and insure anything worth more than the courier's standard cover.",
     old="ship with DHL Express",
     new="ship with UPS (not DHL Express)",
     direct="Change our shipping rule: international parcels now go with UPS instead of DHL Express.",
     passing="Also, we've switched couriers: international parcels go with UPS now, not DHL Express. Please keep that going forward.")

rule("R25", "Surveying customers",
     "NPS and satisfaction surveys, feedback requests and requests for customer interviews",
     "Send each customer at most one survey a quarter, with no more than five questions.",
     old="at most one survey a quarter",
     new="at most one survey every six months",
     direct="Change our rule on surveying customers: send each customer at most one survey every six months instead of one a quarter.",
     passing="And from now on each customer gets at most one survey every six months, not one a quarter. Please make that stick.")

rule("R26", "Public events",
     "event pages and announcements for webinars, meetups and launch events",
     "Every event page and announcement links to the code of conduct and gives events@fernhill.com as the contact for accessibility requests.",
     old="accessibility contact events@fernhill.com",
     new="accessibility contact access@fernhill.com",
     direct="Change our rule on public events: the accessibility contact is now access@fernhill.com instead of events@fernhill.com.",
     passing="Also, accessibility requests now go to access@fernhill.com, not events@. Please keep that for the future.")

rule("R27", "Notices about our service",
     "outage notices, planned maintenance notices, incident reports and announcements of changes to the service",
     "Post it on the status page first, then email the affected customers within one hour.",
     old="email affected customers within one hour",
     new="email affected customers within 30 minutes",
     direct="Change our rule on notices about our service: email the affected customers within 30 minutes instead of one hour.",
     passing="By the way, affected customers now have to be emailed within 30 minutes, not an hour. Please make sure that sticks.")

rule("R28", "Payroll",
     "running payroll, payslips, bonuses and pay corrections",
     "Run payroll on the 25th of the month, or on the last working day before it when the 25th falls at a weekend.",
     old="payroll runs on the 25th",
     new="payroll runs on the 28th",
     direct="Change our payroll rule: payroll now runs on the 28th instead of the 25th, or the last working day before it.",
     passing="Also, payday moves to the 28th from now on instead of the 25th. Please keep that going forward.")


# ---------------------------------------------------------------------------
# Folders and procedures
# ---------------------------------------------------------------------------

folder("email", "Email",
       "Draft, send, forward and reply to internal email, and look up people's addresses. Use when writing to colleagues or answering a message.")
proc("email", "draft_email", "to: list[str], subject: str, points: list[str]", "str",
     "Draft an internal email to colleagues from a list of points and save it as a draft. Returns the draft id.",
     "Write up a note to Sam and Priya with the three decisions from this morning's planning meeting, and leave it as a draft.")
proc("email", "send_draft", "draft_id: str", "None",
     "Send an email draft that is waiting in the outbox.",
     "The note to the ops team is sitting in drafts as d_771; go ahead and send it.")
proc("email", "reply_to_thread", "thread_id: str, points: list[str]", "str",
     "Draft a reply at the end of an internal email thread. Returns the draft id.",
     "Answer the last message in thread th_310 to say the team can do Thursday afternoon.")
proc("email", "forward_email", "message_id: str, to: list[str], note: str", "str",
     "Forward an email to colleagues with a short note on top. Returns the draft id.",
     "Pass message m_5512 on to Lena in finance with a line asking her to check the numbers.")
proc("email", "lookup_contact", "name: str", "dict",
     "Find a person in the address book by name and return their email address, company and role.",
     "What's the email address for Priya, the operations lead at Brightline Foods?")
proc("email", "set_email_signature", "text: str", "None",
     "Replace the signature added to the end of your outgoing email.",
     "Change my email sign-off to include my new job title, Head of Operations.", hidden=True)
proc("email", "archive_old_threads", "older_than_days: int", "int",
     "Archive inbox threads older than the given number of days. Returns how many were archived.",
     "My inbox is a mess; archive everything older than 90 days.", hidden=True)

folder("calendar", "Calendar",
       "Schedule, move and cancel meetings and find free time in people's calendars. Use when arranging a meeting or checking availability.")
proc("calendar", "schedule_internal_meeting", "attendees: list[str], start: str, minutes: int, agenda: str", "str",
     "Create a meeting with colleagues inside the company and invite them. Returns the event id.",
     "Set up a 45-minute retro with the platform team on Wednesday at 2pm.")
proc("calendar", "schedule_external_meeting", "company: str, attendees: list[str], start: str, minutes: int, agenda: str", "str",
     "Create a meeting with people from another company and send them the invitation, for outside meetings that no more specific procedure covers. Returns the event id.",
     "Set up an hour next week with our accountants at Moss & Pike to go through the year-end questions.",
     rules=("R03", "R18"))
proc("calendar", "reschedule_external_meeting", "event_id: str, new_start: str", "None",
     "Move an existing meeting with people outside the company to a new time and send them the updated invitation.",
     "Move our call with the Brightline Foods team (event ev_204) to a later day this week.",
     rules=("R03", "R18"))
proc("calendar", "find_free_slot", "people: list[str], minutes: int, within_days: int", "list[str]",
     "Find times when all the given colleagues are free for a meeting of this length.",
     "When are Sam, Lena and Joe all free for an hour in the next five days?")
proc("calendar", "cancel_meeting", "event_id: str, reason: str", "None",
     "Cancel a calendar event and tell everyone invited why.",
     "Cancel tomorrow's budget sync with the finance team (ev_310); the numbers aren't ready.")
proc("calendar", "set_out_of_office", "start: str, end: str, message: str", "None",
     "Turn on an automatic away reply and block your calendar for the dates.",
     "I'm off from 20 to 24 October; put up an away message and block my calendar.", hidden=True)
proc("calendar", "block_focus_time", "days: list[str], start: str, minutes: int", "None",
     "Block recurring focus time in your own calendar on the given weekdays.",
     "Protect my mornings: block 9 to 11 every Tuesday and Thursday for deep work.", hidden=True)

folder("slides", "Slide decks",
       "Build slide decks for meetings, investors and prospects, add chart slides and export decks as PDF. Use when making or changing a presentation.")
proc("slides", "build_deck", "title: str, sections: list[str]", "str",
     "Build a plain internal slide deck with a title slide and one slide per section. Returns the deck id.",
     "Make a short deck for Monday's all-hands with sections on hiring, the office move and the roadmap.")
proc("slides", "add_chart_slide", "deck_id: str, title: str, series: dict", "None",
     "Add a slide with a line chart of the given series to an existing deck.",
     "Add a chart of weekly active users over the last eight weeks to deck dk_9.")
proc("slides", "export_deck_pdf", "deck_id: str", "str",
     "Export a deck as a PDF file and return its path.",
     "I need deck dk_9 as a PDF to attach to an email.")
proc("slides", "build_investor_deck", "quarter: str, highlights: list[str]", "str",
     "Build the deck for the quarterly meeting with our investors from the quarter's highlights. Returns the deck id.",
     "Put together the deck for this quarter's investor meeting.",
     rules=("R01",))
proc("slides", "build_sales_deck", "prospect: str, plan: str", "str",
     "Build a sales deck for a prospect, with our product's features, the plan's price and reference customers. Returns the deck id.",
     "Make a pitch deck for Kestrel Health showing the Growth plan, its price and a couple of reference customers.",
     rules=("R02", "R04", "R06"))
proc("slides", "build_pricing_review_deck", "plans: list[str]", "str",
     "Build an internal deck comparing our plans and what each costs, for a pricing review. Returns the deck id.",
     "Make a few slides for the internal pricing review comparing our three plans and what each costs.",
     rules=("R04",))
proc("slides", "add_customer_logo_slide", "deck_id: str, customers: list[str]", "None",
     "Add a slide of customer logos, each with the customer's name, to a deck we show outside the company.",
     "Add a slide with the logos of our five biggest customers to the sales deck dk_88.",
     rules=("R02",), hidden=True)
proc("slides", "translate_deck", "deck_id: str, language: str", "str",
     "Make a copy of a deck translated into another language. Returns the new deck id.",
     "We need the onboarding deck dk_41 in Portuguese for the Lisbon team.", hidden=True)

folder("website", "Website and blog",
       "Publish posts on the company blog and case studies, and update pages on the website. Use when putting something on our site.")
proc("website", "publish_blog_post", "title: str, body: str, images: list[str]", "str",
     "Publish a post, with its images, on the company blog. Returns the post's URL.",
     "Publish a post on our blog about the warehouse rollout we just finished with Harbour Logistics, with the two photos from the site.",
     rules=("R02", "R05", "R06"))
proc("website", "update_website_page", "page: str, content: str, images: list[str]", "None",
     "Replace the text and images of a page on the company website.",
     "Refresh the About page on our website with the new team photo and the updated company description.",
     rules=("R05", "R06"))
proc("website", "publish_case_study", "customer: str, story: str, images: list[str]", "str",
     "Publish a case study about a customer's results as a page on the website, with its images. Returns its URL.",
     "Turn the Brightline Foods results into a case study on our site, with the photo of their depot.",
     rules=("R02", "R05", "R06"))
proc("website", "fix_broken_links", "section: str", "int",
     "Find and fix broken links in one section of the website. Returns how many were fixed.",
     "Some links in the help centre are dead; sort them out.", hidden=True)
proc("website", "redirect_url", "old_path: str, new_path: str", "None",
     "Make an old website address send visitors to a new one.",
     "Point /pricing-2025 at the new /pricing page.", hidden=True)

folder("social", "Social media",
       "Post, schedule and reply on the company's social accounts. Use when something should go out on LinkedIn, X or Instagram.")
proc("social", "post_social_update", "channel: str, text: str", "str",
     "Post a short text update to one of the company's social accounts now. Returns the post's URL.",
     "Announce on LinkedIn that Brightline Foods has become our hundredth customer.",
     rules=("R02", "R23"))
proc("social", "schedule_social_post", "channel: str, text: str, publish_at: str", "str",
     "Schedule a text post on one of the company's social accounts for a later time. Returns the scheduled post's id.",
     "Queue a post on X for next Tuesday morning about our new Lisbon office.",
     rules=("R02", "R23"))
proc("social", "reply_to_social_comment", "comment_id: str, text: str", "str",
     "Reply publicly to a comment on one of the company's social posts. Returns the reply's URL.",
     "Someone asked under our LinkedIn post (comment c_8812) whether we work with frozen-food warehouses; answer them.",
     rules=("R02", "R23"))
proc("social", "delete_social_post", "post_id: str", "None",
     "Take down a post from one of the company's social accounts.",
     "Take down the X post p_9921; it has the wrong date.", hidden=True)
proc("social", "report_social_metrics", "period: str", "dict",
     "Count followers, reach and engagement on the company's social accounts over a period.",
     "How did our LinkedIn and X accounts do last month in reach and engagement?", hidden=True)
proc("social", "update_social_bio", "channel: str, text: str", "None",
     "Change the bio text on one of the company's social profiles.",
     "Update our Instagram bio to mention the Lisbon office.", hidden=True)

folder("marketing", "Marketing email",
       "Send the newsletter and email campaigns to our mailing lists. Use for marketing email to customers and subscribers.")
proc("marketing", "send_newsletter", "subject: str, sections: list[str], images: list[str]", "str",
     "Send the newsletter, with its images, to the subscriber list. Returns the campaign id.",
     "Send out October's newsletter with the Lisbon office news, the new release and the photos from the meetup.",
     rules=("R02", "R05", "R06"))
proc("marketing", "send_campaign_email", "segment: str, subject: str, body: str", "str",
     "Send a marketing email to one segment of the mailing list. Returns the campaign id.",
     "Email the trial users in the retail segment about the new inventory forecasting feature.",
     rules=("R06",))
proc("marketing", "export_mailing_list", "list_name: str", "str",
     "Export a mailing list, with subscribers' names and email addresses, as a CSV file. Returns its path.",
     "Export the newsletter subscriber list as a CSV.",
     rules=("R12",), hidden=True)
proc("marketing", "add_subscribers", "list_name: str, emails: list[str]", "int",
     "Add people to a mailing list. Returns how many were added.",
     "Add the twelve people who signed up at the Lisbon meetup to the newsletter list.", hidden=True)
proc("marketing", "remove_bounced_addresses", "list_name: str", "int",
     "Remove the addresses that bounce from a mailing list. Returns how many were removed.",
     "Clean the bounced addresses out of the product updates list.", hidden=True)
proc("marketing", "pause_campaign", "campaign_id: str", "None",
     "Stop a scheduled marketing email from going out.",
     "Hold campaign c_310; the discount code in it is wrong.")

folder("press", "Press and media",
       "Write press releases, answer journalists, pitch stories and keep the media list. Use when dealing with the press.")
proc("press", "draft_press_release", "headline: str, points: list[str]", "str",
     "Draft a press release from a headline and key points. Returns the document id.",
     "Draft a press release announcing our partnership with Orbit Retail.",
     rules=("R02", "R06", "R07"))
proc("press", "reply_to_journalist", "journalist: str, outlet: str, points: list[str]", "str",
     "Draft a reply to a journalist's question. Returns the draft id.",
     "A reporter from Freight Weekly asked how our route forecasting works; draft our answer.",
     rules=("R07",))
proc("press", "pitch_story", "outlet: str, angle: str", "str",
     "Draft a pitch offering a story to a news outlet. Returns the draft id.",
     "Pitch the story of our Lisbon expansion to TechCrunch.",
     rules=("R07",))
proc("press", "update_media_list", "entries: list[dict]", "int",
     "Add or update journalists in the media list. Returns how many changed.",
     "Add the three reporters we met at the logistics summit to our media list.",
     rules=("R07",))
proc("press", "schedule_media_interview", "journalist: str, spokesperson: str, start: str", "str",
     "Arrange an interview between a journalist and one of our spokespeople, and send the invitation. Returns the event id.",
     "Set up a time for the Logistics Today reporter to interview our CEO next week.",
     rules=("R03", "R07", "R18"), hidden=True)
proc("press", "track_press_coverage", "since: str", "list[dict]",
     "List the articles that have mentioned the company since a date.",
     "Find every article that's mentioned us since September.", hidden=True)

folder("investors", "Investor relations",
       "Write updates for our investors, post on the investor portal and answer investors' questions. Use for anything for our investors.")
proc("investors", "draft_investor_update", "month: str, highlights: list[str]", "str",
     "Draft the monthly update email to our investors from the month's highlights. Returns the draft id.",
     "Draft this month's update to our investors: four engineers hired and the Lisbon office open.",
     rules=("R01",))
proc("investors", "post_investor_portal_update", "quarter: str, letter: str", "str",
     "Post the quarterly letter to our investors on the investor portal. Returns its URL.",
     "Post this quarter's letter for our investors on their portal.",
     rules=("R01",))
proc("investors", "answer_investor_question", "investor: str, question: str, points: list[str]", "str",
     "Draft an answer to a question from one of our investors. Returns the draft id.",
     "Anya at Northgate Ventures asked why customer churn went up in the summer; draft our answer.",
     rules=("R01",))
proc("investors", "schedule_investor_call", "investor: str, start: str", "str",
     "Arrange a call with one of our investors and send the invitation. Returns the event id.",
     "Book a call with Anya from Northgate Ventures for next Thursday.",
     rules=("R03", "R18"))
proc("investors", "update_cap_table", "changes: list[dict]", "None",
     "Record new share issues, option grants or transfers in the cap table.",
     "Record the 20,000 options granted to Lena last week in the cap table.", hidden=True)
proc("investors", "list_investor_contacts", "", "list[dict]",
     "List our contacts at each investor, with their fund and email address.",
     "Who are our contacts at each of our investors?", hidden=True)

folder("board", "Board",
       "Prepare board meetings: the board deck, pre-reads, minutes and written resolutions. Use for anything for the board of directors.")
proc("board", "build_board_deck", "meeting_date: str, topics: list[str]", "str",
     "Build the slides for a board meeting from its agenda topics, one slide per topic with the decision asked of the board. Returns the deck id.",
     "Build the deck for the 14 November board meeting covering hiring, the Lisbon office and next year's plan.",
     rules=("R19",))
proc("board", "write_board_minutes", "meeting_date: str, notes: str", "str",
     "Write up the minutes of a board meeting from notes. Returns the document id.",
     "Turn my notes from yesterday's board meeting into minutes.",
     rules=("R19",))
proc("board", "circulate_board_preread", "meeting_date: str, documents: list[str]", "None",
     "Send the pre-read documents for a board meeting to the directors.",
     "Get the pre-reads for next week's board meeting out to the directors.",
     rules=("R19",))
proc("board", "draft_written_resolution", "subject: str, text: str", "str",
     "Draft a written resolution for the directors to sign instead of meeting. Returns the document id.",
     "Draft a written resolution for the board to approve opening the Lisbon subsidiary.",
     rules=("R19",))
proc("board", "track_board_actions", "meeting_date: str", "list[dict]",
     "List the actions the board gave management at a meeting, and which are still open.",
     "What actions did the board give us at the last meeting, and which are still open?", hidden=True)
proc("board", "update_director_register", "changes: list[dict]", "None",
     "Record directors joining or leaving in the statutory register of directors.",
     "Record that Anya Patel joined the board on 1 October.", hidden=True)

folder("billing", "Billing",
       "Create invoices, chase unpaid ones, and issue refunds and credit notes. Use when billing a customer or handling a customer's payment.")
proc("billing", "create_invoice", "customer: str, items: list[dict]", "str",
     "Create an invoice for a customer from line items of description, quantity and unit price. Returns the invoice id.",
     "Invoice Harbour Logistics for 40 hours of onboarding at 150 an hour.",
     rules=("R04",))
proc("billing", "send_payment_reminder", "invoice_id: str", "None",
     "Email a customer a reminder about an unpaid invoice, with the amount due.",
     "Invoice INV-1190 is two weeks overdue; nudge the customer.",
     rules=("R04",))
proc("billing", "issue_refund", "invoice_id: str, amount: float, reason: str", "str",
     "Refund part or all of a paid invoice to the customer. Returns the refund id.",
     "Refund Brightline Foods 300 on invoice INV-1182; they were charged twice for one seat.",
     rules=("R04", "R09"))
proc("billing", "issue_credit_note", "customer: str, amount: float, reason: str", "str",
     "Issue a credit note that reduces a customer's next bill. Returns the credit note id.",
     "Give Orbit Retail a credit of 450 against their next bill for last week's outage.",
     rules=("R04", "R09"))
proc("billing", "convert_currency", "amount: float, from_currency: str, to_currency: str", "float",
     "Convert an amount from one currency to another at today's rate.",
     "How much is 12,500 Swiss francs in pounds today?", hidden=True)
proc("billing", "update_billing_details", "customer: str, details: dict", "None",
     "Change a customer's billing address, billing contact or purchase order number.",
     "Kestrel Health has a new billing address and a PO number for invoices; update their account.", hidden=True)

folder("payables", "Paying suppliers",
       "Pay supplier and contractor invoices, schedule payments and send wire transfers. Use when we owe someone money.")
proc("payables", "pay_vendor_invoice", "vendor: str, invoice_ref: str, amount: float", "str",
     "Pay a supplier's invoice. Returns the payment id.",
     "Pay Greenleaf Supplies' invoice GL-2210 for the office plants.",
     rules=("R10",))
proc("payables", "schedule_supplier_payment", "vendor: str, amount: float, date: str", "str",
     "Schedule a payment to a supplier for a later date. Returns the payment id.",
     "Set up the payment to Moss & Pike for their audit fee to go out on the 30th.",
     rules=("R10",))
proc("payables", "send_wire_transfer", "beneficiary: str, amount: float, currency: str", "str",
     "Send money by wire transfer to a company's bank account. Returns the transfer id.",
     "Wire the deposit to the Lisbon landlord, Casa Alfama Lda, today.",
     rules=("R10",))
proc("payables", "pay_contractor_invoice", "contractor: str, invoice_ref: str, amount: float", "str",
     "Pay an invoice from a freelance contractor. Returns the payment id.",
     "Pay Joe Alvarez's invoice for the October design work.",
     rules=("R10",))
proc("payables", "add_new_vendor", "name: str, bank_details: dict", "str",
     "Set up a new supplier, with their bank details, so that we can pay them. Returns the vendor id.",
     "Set up Brightwave Cleaning as a supplier so we can pay them.",
     rules=("R10",), hidden=True)
proc("payables", "list_unpaid_bills", "", "list[dict]",
     "List the supplier bills we have not paid yet, with their due dates.",
     "Which supplier bills are still unpaid?", hidden=True)
proc("payables", "dispute_vendor_charge", "vendor: str, invoice_ref: str, reason: str", "str",
     "Raise a dispute with a supplier about a charge on their invoice. Returns the dispute id.",
     "Greenleaf billed us twice for October; raise it with them.", hidden=True)

folder("expenses", "Expenses",
       "Submit and approve staff expense claims and report on spending. Use for out-of-pocket spending by staff.")
proc("expenses", "submit_expense_claim", "employee: str, items: list[dict]", "str",
     "Submit an expense claim for an employee from receipts and items. Returns the claim id.",
     "File my expenses from the Lisbon trip: taxis, two dinners and the train.",
     rules=("R04",))
proc("expenses", "approve_expense_claim", "claim_id: str", "None",
     "Approve an employee's expense claim after checking it against the expense policy.",
     "Approve Lena's Lisbon expense claim, EC-778, if it's within policy.",
     rules=("R04",))
proc("expenses", "export_expense_report", "period: str", "str",
     "Export a spending report of staff expense claims over a period, totalled by team and category. Returns the file's path.",
     "Export all staff spending for Q3 as a report for finance.",
     rules=("R04",))
proc("expenses", "issue_corporate_card", "employee: str, monthly_limit: float", "str",
     "Order a company card for an employee with a monthly limit. Returns the card id.",
     "Lena needs a company card with a monthly limit for her travel.", hidden=True)
proc("expenses", "read_receipt", "image: str", "dict",
     "Read a photo of a receipt and return the merchant, date and total.",
     "What's on this receipt photo from the Lisbon dinner?", hidden=True)

folder("payroll", "Payroll",
       "Run the monthly payroll, produce payslips and pay bonuses. Use for paying staff their salaries.")
proc("payroll", "run_payroll", "month: str", "None",
     "Run the payroll for a month, paying every employee's salary.",
     "Run this month's payroll.",
     rules=("R28",))
proc("payroll", "generate_payslips", "month: str", "list[str]",
     "Produce the payslips for a month for every employee. Returns their file paths.",
     "Produce October's payslips for everyone.",
     rules=("R04", "R28"))
proc("payroll", "pay_bonus", "employee: str, amount: float, reason: str", "None",
     "Pay an employee a one-off bonus through payroll.",
     "Pay Sam the bonus we agreed for the migration project.",
     rules=("R28",))
proc("payroll", "correct_employee_pay", "employee: str, month: str, correction: float", "None",
     "Correct an employee's pay for a month and record the change in their HR record.",
     "Joe was underpaid in September by two days of overtime; fix it.",
     rules=("R13", "R28"), hidden=True)
proc("payroll", "update_tax_code", "employee: str, code: str", "None",
     "Change the tax code in an employee's HR record.",
     "HMRC sent Lena a new tax code, 1257L; update it.",
     rules=("R13",), hidden=True)
proc("payroll", "export_payroll_cost_report", "month: str", "str",
     "Export a spending report of payroll costs for a month, by team. Returns the file's path.",
     "How much did payroll cost each team in September? I need it as a report.",
     rules=("R04",), hidden=True)

folder("hr", "People and HR",
       "Keep employee records up to date: salary changes, leave, performance reviews and leavers. Use for HR admin about current staff.")
proc("hr", "update_salary", "employee: str, new_salary: float, effective: str", "None",
     "Record a change to an employee's salary in their HR record.",
     "Lena's salary goes up from 1 November as agreed in her review; record it.",
     rules=("R13",))
proc("hr", "record_leave", "employee: str, start: str, end: str, kind: str", "None",
     "Record an employee's leave, such as holiday, sickness or parental leave, in their HR record.",
     "Sam is on parental leave from 3 November to 2 February; log it.",
     rules=("R13",))
proc("hr", "file_performance_review", "employee: str, review: str", "None",
     "File an employee's performance review in their HR record.",
     "File Joe's mid-year review from the notes his manager sent.",
     rules=("R13",))
proc("hr", "process_leaver", "employee: str, last_day: str", "None",
     "Process an employee who is leaving: close their HR record and revoke their access to systems and the building.",
     "Mark is leaving on 31 October; process his departure.",
     rules=("R13", "R16"))
proc("hr", "send_offer_letter", "candidate: str, role: str, salary: float, start: str", "str",
     "Send a job candidate our offer letter for a role. Returns the letter's id.",
     "Send Ana Costa the offer for the senior designer role, starting 1 December.",
     rules=("R11",), hidden=True)
proc("hr", "export_employee_list", "fields: list[str]", "str",
     "Export a list of employees with the chosen fields, such as name, team and email, as a CSV file. Returns its path.",
     "Give me a spreadsheet of everyone's name, team and work email.",
     rules=("R12",), hidden=True)
proc("hr", "update_org_chart", "changes: list[dict]", "None",
     "Change who reports to whom in the org chart.",
     "Move Sam under Lena in the org chart.", hidden=True)

folder("recruiting", "Recruiting",
       "Post job ads, schedule interviews and keep candidates moving through the hiring pipeline. Use for hiring new people.")
proc("recruiting", "post_job_ad", "role: str, description: str", "str",
     "Post a job ad on the careers page of our website. Returns its URL.",
     "Advertise the senior designer role on our careers page.",
     rules=("R06",))
proc("recruiting", "schedule_interview", "candidate: str, interviewers: list[str], start: str", "str",
     "Schedule an interview with a job candidate and send them the invitation. Returns the event id.",
     "Set up a first interview for Ana Costa with Sam and Lena on Tuesday.",
     rules=("R03", "R11", "R18"))
proc("recruiting", "reject_candidate", "candidate: str, role: str", "None",
     "Tell a job candidate that we are not moving forward with them.",
     "Let Tom Reyes know we won't be moving forward for the data engineer role.",
     rules=("R11",))
proc("recruiting", "follow_up_candidate", "candidate: str, note: str", "None",
     "Send a job candidate a follow-up message about where their application stands.",
     "Check in with Ana Costa about whether she's had a chance to look at the take-home task.",
     rules=("R11",))
proc("recruiting", "export_candidate_pipeline", "role: str", "str",
     "Export the candidates for a role, with their stage and contact details, as a CSV file. Returns its path.",
     "Export the pipeline for the data engineer role, with each candidate's stage and contact details.",
     rules=("R12",), hidden=True)
proc("recruiting", "record_interview_feedback", "candidate: str, interviewer: str, feedback: str", "None",
     "Record an interviewer's feedback on a candidate in the hiring system.",
     "Log Lena's feedback from Ana Costa's design interview.", hidden=True)
proc("recruiting", "check_references", "candidate: str, referees: list[str]", "None",
     "Contact a candidate's referees to confirm their past employment.",
     "Contact Ana Costa's two referees to confirm her time at Studio Norte.", hidden=True)

folder("sales", "Sales",
       "Manage deals in the CRM, book product demos, send quotes and order forms, and forecast the pipeline. Use for selling to prospects.")
proc("sales", "create_quote", "prospect: str, plan: str, seats: int", "str",
     "Create a price quote for a prospect for a plan and number of seats, and send it to them. Returns the quote id.",
     "Send Kestrel Health a quote for 40 seats on the Growth plan.",
     rules=("R04",))
proc("sales", "book_demo", "prospect: str, attendees: list[str], start: str", "str",
     "Book a product demo with a prospect and send them the invitation. Returns the event id.",
     "Book a product demo with the buyers at Orbit Retail for next Wednesday.",
     rules=("R03", "R18"))
proc("sales", "update_deal_stage", "deal_id: str, stage: str", "None",
     "Move a deal to a new stage in the CRM.",
     "Move the Kestrel Health deal to negotiation.")
proc("sales", "log_sales_call", "deal_id: str, notes: str", "None",
     "Log notes from a sales call against a deal in the CRM.",
     "Log my notes from today's call with Quarry & Co against their deal.")
proc("sales", "forecast_pipeline", "quarter: str", "dict",
     "Forecast how much of the sales pipeline will close in a quarter.",
     "What will we close this quarter? Give me the pipeline forecast.",
     rules=("R04", "R20"))
proc("sales", "send_order_form", "customer: str, plan: str, seats: int", "str",
     "Send a customer the order form, the short contract they sign to buy a plan, with its price. Returns the document id.",
     "Kestrel Health agreed to the Growth plan for 40 seats; send them the paperwork to sign.",
     rules=("R04", "R08"))
proc("sales", "export_contacts", "segment: str", "str",
     "Export the CRM's contacts in a segment, with names, emails and phone numbers, as a CSV file. Returns its path.",
     "Export all our contacts at retail prospects with their emails and phone numbers.",
     rules=("R12",), hidden=True)
proc("sales", "enrich_lead", "company: str", "dict",
     "Look up a prospect company's size, industry and the tools it uses.",
     "Find out how big Quarry & Co is and what they use for logistics today.", hidden=True)

folder("customer_success", "Customer success",
       "Run business reviews and health checks for existing customers, survey them and handle cancellations. Use for looking after accounts after the sale.")
proc("customer_success", "schedule_business_review", "customer: str, start: str", "str",
     "Schedule a quarterly business review with a customer and send them the invitation. Returns the event id.",
     "Set up Brightline Foods' quarterly business review for early next month.",
     rules=("R03", "R18"))
proc("customer_success", "cancel_subscription", "customer: str, reason: str, prorated_refund: bool", "None",
     "Cancel a customer's subscription, with a prorated refund of the unused time when asked.",
     "Orbit Retail is closing their account at the end of the month; cancel it and refund the unused time.",
     rules=("R04", "R09"))
proc("customer_success", "send_nps_survey", "customers: list[str]", "str",
     "Send an NPS survey to a group of customers. Returns the survey id.",
     "Send this quarter's NPS survey to our enterprise customers.",
     rules=("R25",))
proc("customer_success", "request_customer_interview", "customer: str, topic: str", "None",
     "Ask a customer whether someone would talk to our product team about a topic.",
     "Ask Harbour Logistics if someone would talk to our product team about the new forecasting feature.",
     rules=("R25",), hidden=True)
proc("customer_success", "check_account_health", "customer: str", "dict",
     "Summarise a customer account's health: usage, open tickets and renewal date.",
     "How healthy is the Quarry & Co account: usage, tickets, renewal date?")
proc("customer_success", "respond_to_complaint", "customer: str, complaint: str, points: list[str]", "str",
     "Draft a reply to a customer's complaint. Returns the draft id.",
     "Brightline's COO wrote in angry about last week's missed delivery plans; draft our reply.",
     rules=("R17",), hidden=True)
proc("customer_success", "export_account_usage", "customer: str", "str",
     "Export a customer account's usage totals for the last 90 days (routes planned, warehouses, active seats). Returns the file's path.",
     "Get me Kestrel Health's product usage for the last 90 days.", hidden=True)
proc("customer_success", "ship_scanner_kit", "customer: str, address: str", "str",
     "Ship a kit of barcode scanners to a customer's warehouse. Returns the tracking number.",
     "Orbit Retail's new warehouse in Rotterdam needs a scanner kit; send one.",
     rules=("R24",), hidden=True)

folder("support", "Customer support",
       "Answer, escalate and close support tickets, and search the help centre. Use for customers' questions and problems.")
proc("support", "reply_to_ticket", "ticket_id: str, points: list[str]", "str",
     "Reply to a customer on a support ticket. Returns the reply id.",
     "Answer ticket T-4410: the customer can't export their delivery report.",
     rules=("R17",))
proc("support", "escalate_ticket", "ticket_id: str, team: str, reason: str", "None",
     "Hand a support ticket to another internal team with a note on why.",
     "Ticket T-4412 looks like a bug in the routing engine; pass it to engineering.")
proc("support", "close_ticket", "ticket_id: str, resolution: str", "None",
     "Close a support ticket and record how it was resolved.",
     "Close T-4398; the customer confirmed it's fixed.")
proc("support", "search_help_center", "query: str", "list[dict]",
     "Search the help centre's articles.",
     "Do we have a help article that explains how to set delivery windows?")
proc("support", "send_csat_survey", "ticket_id: str", "None",
     "Send the customer on a ticket a short satisfaction survey.",
     "Ask the customer on T-4398 how we did.",
     rules=("R25",), hidden=True)
proc("support", "merge_tickets", "ticket_ids: list[str]", "str",
     "Merge duplicate support tickets into one. Returns the remaining ticket's id.",
     "T-4415 and T-4416 are the same problem from the same customer; combine them.", hidden=True)
proc("support", "export_ticket_requesters", "period: str", "str",
     "Export everyone who opened a ticket in a period, with their name and email, as a CSV file. Returns its path.",
     "List everyone who opened a ticket last month, with their email, as a CSV.",
     rules=("R12",), hidden=True)

folder("legal", "Contracts",
       "Send NDAs and customer contracts, renew them and keep the contract register. Use for legal paperwork with other companies.")
proc("legal", "send_nda", "company: str, contact: str", "str",
     "Send a company our mutual NDA to sign. Returns the document id.",
     "Send an NDA to Quarry & Co before we share the roadmap.",
     rules=("R08",))
proc("legal", "send_customer_contract", "customer: str, term_years: int", "str",
     "Send a customer our master services agreement for a multi-year deal. Returns the document id.",
     "Kestrel Health wants a three-year master agreement; send it over.",
     rules=("R08",))
proc("legal", "renew_contract", "contract_id: str, years: int", "str",
     "Renew an existing contract for a further term. Returns the renewal's id.",
     "Renew the Harbour Logistics agreement for another two years.",
     rules=("R08",))
proc("legal", "update_contract_register", "contract_id: str, fields: dict", "None",
     "Update a contract's entry in the contract register, such as its expiry date or owner.",
     "Record in the contract register that the Orbit Retail NDA expires next March.")
proc("legal", "look_up_clause", "contract_id: str, clause: str", "str",
     "Return the text of one clause of a signed contract.",
     "What does the liability cap in the Harbour Logistics contract say?", hidden=True)
proc("legal", "file_trademark", "mark: str, classes: list[int]", "str",
     "Start a trademark application. Returns the application's reference.",
     "Start a trademark application for the name 'Routefinder' in the software classes.", hidden=True)

folder("travel", "Travel",
       "Book flights, trains and hotels and send itineraries. Use when someone travels for work.")
proc("travel", "book_flight", "traveller: str, origin: str, destination: str, date: str", "str",
     "Book a flight for someone travelling for the company. Returns the booking reference.",
     "Book Lena a flight from London to Lisbon on 12 November.",
     rules=("R14",))
proc("travel", "book_hotel", "traveller: str, city: str, check_in: str, nights: int", "str",
     "Book a hotel for someone travelling for the company. Returns the booking reference.",
     "Find Sam a hotel in Lisbon for three nights from the 12th.",
     rules=("R14",))
proc("travel", "book_train", "traveller: str, origin: str, destination: str, date: str", "str",
     "Book a train for someone travelling for the company. Returns the booking reference.",
     "Get Joe a train from London to Paris on Monday morning.",
     rules=("R14",))
proc("travel", "book_hire_car", "traveller: str, city: str, start: str, days: int", "str",
     "Book a hire car for someone travelling for the company. Returns the booking reference.",
     "Joe needs a car in Porto for two days next week.",
     rules=("R14",), hidden=True)
proc("travel", "send_itinerary", "traveller: str, trip_id: str", "None",
     "Email a traveller their full itinerary for a trip.",
     "Send Lena her full itinerary for the Lisbon trip.")
proc("travel", "cancel_booking", "booking_ref: str", "None",
     "Cancel a travel booking.",
     "Lena's not going to Lisbon after all; cancel booking BK-5521.", hidden=True)
proc("travel", "request_visa_letter", "traveller: str, country: str", "str",
     "Write an invitation letter that a visitor can use for a visa. Returns the document id.",
     "Ana needs an invitation letter for her visa to visit our London office.", hidden=True)

folder("procurement", "Buying",
       "Raise purchase orders, buy and renew software subscriptions and order office supplies. Use when the company buys something.")
proc("procurement", "raise_purchase_order", "vendor: str, items: list[dict]", "str",
     "Raise a purchase order with a supplier for a list of items. Returns the PO number.",
     "Raise a PO for 12 standing desks from Fjord Office.",
     rules=("R04", "R22"))
proc("procurement", "buy_software_subscription", "product: str, seats: int", "str",
     "Buy a new subscription to a software product for a number of seats. Returns the order id.",
     "Get us 15 seats of Figma.",
     rules=("R22",))
proc("procurement", "renew_tool_subscription", "product: str", "str",
     "Renew a software subscription we already pay for. Returns the order id.",
     "Our Notion subscription ends next week; renew it.",
     rules=("R22",))
proc("procurement", "order_office_supplies", "items: list[str]", "str",
     "Order office supplies such as paper, coffee and stationery. Returns the order id.",
     "We're out of printer paper and coffee; order more.",
     rules=("R22",))
proc("procurement", "sign_vendor_agreement", "vendor: str, agreement_id: str", "None",
     "Sign a supplier's agreement with them.",
     "Fjord Office sent their supply agreement; get it signed.",
     rules=("R08",), hidden=True)
proc("procurement", "cancel_software_subscription", "product: str", "None",
     "Cancel a software subscription we no longer use.",
     "We don't use Loom any more; stop paying for it.", hidden=True)
proc("procurement", "compare_supplier_quotes", "quotes: list[dict]", "dict",
     "Compare quotes from several suppliers and say which is the best value.",
     "Which of these three cleaning quotes is the best value?",
     rules=("R04",), hidden=True)

folder("analytics", "Reports and dashboards",
       "Export reports of business metrics and refresh dashboards. Use when someone needs numbers.")
proc("analytics", "export_revenue_report", "period: str", "str",
     "Export a report of revenue by month over a period. Returns the file's path.",
     "Export monthly revenue for the last twelve months as a report for the finance team.",
     rules=("R04",))
proc("analytics", "export_signup_report", "period: str", "str",
     "Export a report of new sign-ups by week over a period. Returns the file's path.",
     "How many people signed up each week this quarter? Export it.")
proc("analytics", "refresh_dashboard", "name: str", "None",
     "Recompute the numbers on a named analytics dashboard.",
     "The sales dashboard looks stale; get it up to date.")
proc("analytics", "export_user_data", "account: str", "str",
     "Export every personal record we hold about the users of one customer account, for a data access request. Returns the file's path.",
     "Orbit Retail sent a data access request for their users; export everything we hold on them.",
     rules=("R12",), hidden=True)
proc("analytics", "share_dataset_with_partner", "partner: str, dataset: str", "None",
     "Send a dataset to a partner company.",
     "Send Routewise the delivery-times dataset they asked for.",
     rules=("R21",), hidden=True)
proc("analytics", "build_churn_report", "period: str", "str",
     "Build a report of which customers left in a period and why. Returns the file's path.",
     "Which customers churned last quarter, and why?")
proc("analytics", "run_sql_query", "query: str", "list[dict]",
     "Run a read-only SQL query against the analytics warehouse.",
     "Run a quick query to count active warehouses per customer.", hidden=True)

folder("finance", "Finance",
       "Close the books, draft budgets and forecasts, and report profit and loss. Use for company accounts and planning.")
proc("finance", "forecast_revenue", "months: int", "dict",
     "Forecast revenue for the coming months.",
     "Give me a revenue forecast for the next six months.",
     rules=("R04", "R20"))
proc("finance", "project_cash_flow", "months: int", "dict",
     "Project cash in and out for the coming months, and when cash would run out.",
     "When do we run out of cash at the current burn? Project it for the next 18 months.",
     rules=("R04", "R20"))
proc("finance", "draft_budget", "year: int, teams: list[str]", "str",
     "Draft next year's budget for the given teams. Returns the document id.",
     "Start next year's budget for engineering, sales and operations.",
     rules=("R04", "R20"))
proc("finance", "close_month", "month: str", "None",
     "Close the books for a month: check every transaction is posted, then lock the ledger.",
     "Close the books for September.")
proc("finance", "export_pnl", "period: str", "str",
     "Export the profit and loss statement for a period. Returns the file's path.",
     "I need the P&L for Q3.",
     rules=("R04",))
proc("finance", "send_audit_data", "auditor: str, datasets: list[str]", "None",
     "Send our auditors the datasets they ask for, such as the general ledger and bank statements.",
     "Moss & Pike, our auditors, need the general ledger and bank statements for the year.",
     rules=("R21",), hidden=True)
proc("finance", "reconcile_bank_account", "account: str, month: str", "dict",
     "Match a month's bank transactions against the ledger and list what does not match.",
     "Match September's bank transactions against the ledger for the main account.", hidden=True)

folder("it", "IT and accounts",
       "Order laptops, set up accounts and grant or remove access to systems. Use for IT requests from staff.")
proc("it", "grant_system_access", "person: str, system: str, role: str", "None",
     "Give a colleague access to a system or shared drive with a given role.",
     "Give Ana Costa editor access to the design drive.",
     rules=("R16",))
proc("it", "revoke_access", "person: str, system: str", "None",
     "Remove someone's access to a system.",
     "Remove Mark's access to the finance system.",
     rules=("R16",))
proc("it", "order_laptop", "person: str, model: str", "str",
     "Buy a laptop for a colleague. Returns the order id.",
     "Ana starts on 1 December; order her a MacBook.",
     rules=("R22",))
proc("it", "create_email_account", "person: str", "str",
     "Create a work email account for a new colleague. Returns the address.",
     "Set up a work email for Ana Costa.",
     rules=("R16",))
proc("it", "ship_laptop", "person: str, address: str", "str",
     "Ship a laptop to a colleague's address. Returns the tracking number.",
     "Send Ana's new laptop to her home in Porto.",
     rules=("R24",), hidden=True)
proc("it", "reset_password", "person: str, system: str", "None",
     "Reset a colleague's password for a system when they are locked out.",
     "Joe's locked out of the VPN; reset his password.", hidden=True)

folder("security", "Security and compliance",
       "Run access reviews, collect audit evidence and answer security questionnaires. Use for security and compliance work.")
proc("security", "run_access_review", "system: str", "list[dict]",
     "List who has which access to a system, for the quarterly access review.",
     "Who has admin access to the production database? Run the quarterly review.")
proc("security", "collect_audit_evidence", "control: str", "list[str]",
     "Gather the evidence for one control of a compliance audit, such as SOC 2. Returns file paths.",
     "Gather the evidence for the SOC 2 control on employee onboarding.")
proc("security", "answer_security_questionnaire", "customer: str, questionnaire_id: str", "str",
     "Fill in a customer's security questionnaire from our policies. Returns the document id.",
     "Kestrel Health sent a security questionnaire (q_88); fill it in from our policies.")
proc("security", "grant_vendor_access", "vendor: str, system: str", "None",
     "Give a supplier's or partner's engineers access to one of our systems.",
     "Routewise's engineers need read access to our staging environment for the integration.",
     rules=("R16",), hidden=True)
proc("security", "report_phishing", "message_id: str", "None",
     "Report a suspicious email to the security team and remove it from inboxes.",
     "I got a suspicious email pretending to be from our bank (m_6123); report it.", hidden=True)
proc("security", "share_logs_with_vendor", "vendor: str, logs: str", "None",
     "Send logs from our systems to a supplier, for example their support team.",
     "Send Datadog support the error logs from last Tuesday's incident.",
     rules=("R21",), hidden=True)

folder("engineering", "Engineering operations",
       "Deploy services, run database migrations, switch feature flags and roll back bad releases. Use for changes to our production systems.")
proc("engineering", "deploy_service", "service: str, version: str", "None",
     "Deploy a version of a service to production.",
     "Ship version 4.2 of the routing service.",
     rules=("R15",))
proc("engineering", "run_migration", "database: str, migration: str", "None",
     "Apply a database migration to the production database.",
     "Apply the migration that adds delivery windows to the production database.",
     rules=("R15",))
proc("engineering", "toggle_feature_flag", "flag: str, enabled: bool", "None",
     "Turn a feature flag on or off in production.",
     "Turn on the new forecasting feature for everyone.",
     rules=("R15",))
proc("engineering", "rollback_deploy", "service: str", "None",
     "Roll a production service back to its previous version.",
     "The last release of the billing service is throwing errors; put back the previous version.",
     rules=("R15",))
proc("engineering", "update_production_config", "service: str, key: str, value: str", "None",
     "Change a configuration value of a service in production.",
     "Raise the routing service's request timeout to 30 seconds in production.",
     rules=("R15",), hidden=True)
proc("engineering", "create_staging_env", "branch: str", "str",
     "Create a temporary test environment from a code branch. Returns its URL.",
     "Spin up a test environment for the delivery-windows branch.", hidden=True)
proc("engineering", "page_on_call", "service: str, summary: str", "None",
     "Page the engineer on call for a service.",
     "Wake up whoever's on call for the API; it's returning 500s.", hidden=True)

folder("status", "Service status",
       "Post outage and maintenance notices on the status page and email affected customers. Use when the service is down or changing.")
proc("status", "post_outage_notice", "summary: str, affected: list[str]", "str",
     "Post a notice of an ongoing outage on the status page and email the affected customers. Returns the notice id.",
     "Tell customers the routing API has been down since 14:10 and we're working on it.",
     rules=("R18", "R27"))
proc("status", "announce_planned_maintenance", "window: str, impact: str", "str",
     "Announce planned maintenance on the status page and email the affected customers. Returns the notice id.",
     "Let customers know we'll take the dashboard down for maintenance on Sunday from 2 to 4am.",
     rules=("R18", "R27"))
proc("status", "post_incident_report", "incident_id: str", "str",
     "Publish the write-up of a past incident for customers. Returns its URL.",
     "Publish the write-up of last week's delivery-tracking outage (inc_77) for customers.",
     rules=("R27",), hidden=True)
proc("status", "announce_service_change", "change: str, effective: str", "str",
     "Announce a change to the service, such as a retired feature, to customers. Returns the notice id.",
     "Tell customers that from January we're retiring the old v1 API.",
     rules=("R27",))
proc("status", "subscribe_to_status", "email: str", "None",
     "Subscribe an email address to status page alerts.",
     "Add the ops team's address to the status page alerts.", hidden=True)

folder("events", "Events",
       "Publish event pages, announce webinars and handle registrations. Use for webinars, meetups and conferences.")
proc("events", "publish_event_page", "event: str, details: str, images: list[str]", "str",
     "Publish a page for one of our events, with its images. Returns its URL.",
     "Put up a page for our Lisbon meetup on 20 November, with the venue photo.",
     rules=("R05", "R26"))
proc("events", "announce_webinar", "title: str, start: str, speakers: list[str]", "str",
     "Announce an upcoming webinar to our customers and subscribers. Returns the announcement id.",
     "Announce next month's webinar on warehouse forecasting with Lena speaking.",
     rules=("R18", "R26"))
proc("events", "export_attendee_list", "event: str", "str",
     "Export the people registered for an event, with their names and emails, as a CSV file. Returns its path.",
     "Get me the list of people registered for the Lisbon meetup, with their emails.",
     rules=("R12",))
proc("events", "book_speaker_travel", "speaker: str, event: str", "str",
     "Book flights and a hotel for a guest speaker at one of our events. Returns the booking reference.",
     "Our guest speaker for the Lisbon meetup needs flights and a hotel.",
     rules=("R14",), hidden=True)
proc("events", "ship_event_materials", "event: str, items: list[str]", "str",
     "Ship banners, demo kits and other materials to an event venue. Returns the tracking number.",
     "Send the banners and the demo kit to the Lisbon venue.",
     rules=("R24",), hidden=True)
proc("events", "submit_conference_talk", "conference: str, abstract: str", "str",
     "Submit a talk proposal to an outside conference's call for papers. Returns the submission id.",
     "Submit Lena's talk on forecasting to LogiConf's call for papers.", hidden=True)

folder("facilities", "Office and facilities",
       "Book meeting rooms, register visitors, issue building passes and report office repairs. Use for anything about our office.")
proc("facilities", "book_meeting_room", "start: str, minutes: int, people: int", "str",
     "Reserve a meeting room big enough for the group. Returns the room's name.",
     "Find us a room for eight people tomorrow from 2 to 3pm.")
proc("facilities", "register_visitor", "name: str, company: str, date: str", "None",
     "Tell reception that a visitor is coming on a date.",
     "Dana from Harbour Logistics is visiting on Thursday; let reception know.")
proc("facilities", "issue_building_pass", "person: str, expires: str", "str",
     "Issue a pass that lets someone into the office building. Returns the pass number.",
     "Ana needs a building pass from her first day.",
     rules=("R16",))
proc("facilities", "report_office_issue", "issue: str", "str",
     "Report a problem in the office, such as a broken heater, to the building manager. Returns the ticket id.",
     "The heating on the third floor isn't working; report it.")
proc("facilities", "arrange_desk_move", "people: list[str], new_desks: list[str]", "None",
     "Arrange for people to move to new desks in the office.",
     "Move the support team to the desks by the window next Monday.", hidden=True)

folder("partners", "Partnerships",
       "Sign partner agreements, meet partners, give them portal access and register the deals they bring. Use for working with partner companies.")
proc("partners", "sign_partner_agreement", "partner: str, agreement_id: str", "None",
     "Sign a partnership agreement with a partner company.",
     "Routewise's reseller agreement is ready; get it signed.",
     rules=("R08",))
proc("partners", "schedule_partner_meeting", "partner: str, start: str", "str",
     "Schedule a meeting with a partner company and send them the invitation. Returns the event id.",
     "Set up our quarterly sync with the Routewise partnership team.",
     rules=("R03", "R18"))
proc("partners", "give_partner_portal_access", "partner: str, people: list[str]", "None",
     "Give people at a partner company access to our partner portal.",
     "Give the two new Routewise engineers access to our partner portal.",
     rules=("R16",))
proc("partners", "register_partner_deal", "partner: str, customer: str", "None",
     "Record that a partner brought us a customer deal.",
     "Routewise brought in Quarry & Co; register the deal against them.")
proc("partners", "pay_partner_commission", "partner: str, amount: float", "str",
     "Pay a partner company its commission on a deal it brought us. Returns the payment id.",
     "Pay Routewise their commission on the Quarry & Co deal.",
     rules=("R10",), hidden=True)
proc("partners", "list_partner_integrations", "", "list[dict]",
     "List the partners that have live integrations with us.",
     "Which partners have live integrations with us right now?", hidden=True)


def check() -> None:
    """Consistency checks on the catalog (used by the tests)."""
    names = [p["name"] for p in PROCEDURES]
    assert len(names) == len(set(names)), "procedure names must be unique"
    for p in PROCEDURES:
        assert p["folder"] in FOLDERS, p
        for r in p["rules"]:
            assert r in RULES, (p["name"], r)
    for rid in RULES:
        assert any(rid in p["rules"] for p in PROCEDURES), f"{rid} governs nothing"
