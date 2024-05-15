# Databricks notebook source
CATALOG_NAME = "cjc"
SCHEMA_NAME = "ml_serv"

spark.sql("CREATE CATALOG IF NOT EXISTS cjc")
spark.sql("CREATE SCHEMA IF NOT EXISTS cjc.ml_serv")
spark.sql("CREATE VOLUME IF NOT EXISTS cjc.ml_serv.myc")
spark.sql("use cjc.ml_serv")

# COMMAND ----------

import pandas as pd

pii_sentences = [
    "Alice Smith's email is alice.smith@email.com and lives at 123 Maple St, Springfield.",
    "Bob Johnson's phone number is 555-1234, residing at 456 Pine Lane, Lakeside.",
    "Carol White mentioned her SSN is 123-45-6789, currently at 789 Oak Ave, Rivertown.",
    "David Brown's license plate is ABC1234 and his address is 101 Birch Rd, Hilltown.",
    "Eve Davis shared her passport number, G12345678, while living at 202 Cedar St, Coastcity.",
    "Frank Moore's credit card number is 1234 5678 9012 3456, billing to 303 Elm St, Greentown.",
    "Grace Lee's driver's license is L123-4567-8901, with a domicile at 404 Aspen Way, Frostville.",
    "Henry Wilson's bank account number is 123456789, banking at 505 Walnut St, Sunnyvale.",
    "Ivy Young disclosed her birthdate, 01/02/1990, alongside her residence at 606 Pinecone Rd, Raincity.",
    "Jack Taylor's employee ID is 7890, working at 707 Redwood Blvd, Cloudtown.",
    "Kathy Green's insurance policy is INS-123456, covered at 808 Maple Grove, Windyville.",
    "Leo Carter mentioned his membership number, MEM-789123, frequenting 909 Cherry Lane, Starcity.",
    "Mia Ward's patient ID is PAT-456789, consulting at 1010 Willow Path, Moonville.",
    "Nathan Ellis's booking reference is REF1234567, staying at 1111 Ivy Green Rd, Sunnyside.",
    "Olivia Sanchez's pet's name is Whiskers, living together at 1212 Magnolia St, Petville.",
    "Peter Gomez's library card number is 1234567, a patron of 1313 Lilac Lane, Booktown.",
    "Quinn Torres is registered under the voter ID VOT-7890123, residing at 1414 Oakdale St, Voteville.",
    "Rachel Kim mentioned her alumni number, ALU-123789, belonging to 1515 Pine St, Gradtown.",
    "Steve Adams's gym membership is GYM-456123, exercising at 1616 Fir Ave, Muscleville.",
    "Tina Nguyen's loyalty card is LOY-789456, shopping at 1717 Spruce Way, Marketcity."
]

df = spark.createDataFrame(
    pd.DataFrame(
        {
            "unmasked_text": pii_sentences
        }
    )
)
display(df)

# COMMAND ----------

_ = (
  df.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.pii_data")
)

# COMMAND ----------

import pandas as pd

sentences = [
    """
    Alice Smith is a renowned baker in Springfield, known for her artisanal breads and pastries. After studying culinary arts in Paris, she returned to her hometown to open 'Alice's Pantry', a boutique bakery that has become a local favorite. Her commitment to using organic, locally-sourced ingredients has earned her accolades and a devoted clientele. Alice's home on Maple St is often filled with the warm, comforting aroma of freshly baked goods.
    """,
    """
    Bob Johnson is a retired Navy captain who now enjoys a quieter life by the lake. He spends his days restoring old boats and sailing on Lakeside's serene waters. His home on Pine Lane is filled with nautical memorabilia, and he is often found in his workshop, tinkering with his latest project. Bob is a pillar of the community, organizing the annual Lakeside Regatta and mentoring young sailors.
    """,
    """
    Carol White is a social worker dedicated to helping families in Rivertown navigate challenging times. She works tirelessly to ensure that everyone has access to necessary resources, such as counseling, financial aid, and healthcare. Her home on Oak Ave serves as a safe haven for those in need, and she is respected and loved by the community for her empathy and unyielding support.
    """,
    """
    David Brown is a local mechanic known for his expertise in classic cars. His garage on Birch Rd is a treasure trove of vintage models, each lovingly restored by David himself. He often drives his favorite, a cherry-red convertible, in the town parade. David's knowledge and skill are not just limited to cars; he's also the go-to person for any mechanical advice.
    """,
    """
    Eve Davis is an avid traveler and cultural blogger who documents her journeys online. She has visited over fifty countries and shares her experiences and tips with a large following. Her home in Coastcity is decorated with artifacts and photographs from her travels, and she often hosts gatherings where she cooks dishes from around the world for her friends.
    """,
    """
    Frank Moore is a successful entrepreneur who has built his business from the ground up. He owns a chain of electronics stores across the state, with his main office in Greentown. Known for his sharp business acumen and innovative strategies, Frank is a mentor to aspiring young entrepreneurs. His home on Elm St is equipped with the latest tech gadgets, a testament to his passion for his work.
    """,
    """
    Grace Lee is a passionate advocate for environmental conservation. She works with various non-profits to promote sustainable living and protect local wildlife. Her home in Frostville is a model of eco-friendliness, featuring solar panels, a rainwater harvesting system, and a vegetable garden. Grace is often seen leading community clean-up drives or giving talks on environmental awareness.
    """,
    """
    Henry Wilson is a retired banker who has taken up teaching financial literacy in his community. His experiences in the banking industry have equipped him with a wealth of knowledge, which he eagerly shares with others. His home on Walnut St is a popular meeting spot for his students and colleagues, where they discuss everything from investment strategies to economic policies.
    """,
    """
    Ivy Young is a freelance graphic designer who specializes in creating branding for startups. She is known for her creative designs and has won several awards in her field. Her home studio on Pinecone Rd is a hub of creativity, filled with her artworks and design tools. Ivy is also a volunteer at the local arts center, teaching classes to inspire others.
    """,
    """
    Jack Taylor works as a project manager for a large construction firm and is currently overseeing the development of a new shopping center in Cloudtown. He is known for his effective management style and dedication to ensuring all projects are delivered on time and within budget. His home on Redwood Blvd is often the site of barbecue gatherings for his team and family.
    """,
    """
    Kathy Green is a healthcare administrator with a deep commitment to improving patient care. Her work ensures that the healthcare facility in Windyville operates smoothly and that patients receive high-quality care. Kathy's home on Maple Grove is a cozy retreat where she enjoys gardening and hosting dinner parties that feature healthy, homemade meals.
    """,
    """
    Leo Carter is a history professor with a passion for medieval European history. His lectures are well-attended, and he is known for his ability to make history come alive. His home on Cherry Lane is filled with books, manuscripts, and artifacts that he has collected over the years. Leo is also a member of several historical societies and frequently participates in reenactments.
    """,
    """
    Mia Ward is a pediatric nurse who provides compassionate care to her young patients. Her gentle demeanor and positive attitude make her a favorite among children and parents alike. Her home on Willow Path is warmly decorated,

 reflecting her nurturing nature, and she spends her free time volunteering at local schools to promote health education.
    """,
    """
    Nathan Ellis is a travel consultant who helps his clients plan their dream vacations. He has a knack for finding hidden gems and tailoring trips to individual preferences. His home on Ivy Green Rd is filled with travel guides, maps, and photos of his own adventures. Nathan also runs a travel blog where he shares his experiences and tips for stress-free travel.
    """,
    """
    Olivia Sanchez is a veterinary technician who loves all animals, big and small. Her home on Magnolia St is often a temporary shelter for rescued pets waiting for adoption. Olivia's compassionate care ensures that these animals are healthy and happy. She also organizes community pet adoption days and educates pet owners on proper care and nutrition.
    """,
    """
    Peter Gomez is a librarian who has a passion for books and helping people discover their love for reading. His extensive knowledge of literature makes him a valuable resource at the library in Booktown. His home on Lilac Lane is lined with bookshelves, each filled with his favorite novels and rare finds. Peter often hosts book club meetings and author talks.
    """,
    """
    Quinn Torres is an active community organizer who focuses on voter registration and education. Her efforts ensure that everyone in Voteville has the opportunity to vote and understands the importance of their participation. Her home on Oakdale St is the headquarters for her team of volunteers, and she is often seen leading workshops and speaking at community events.
    """,
    """
    Rachel Kim is an alumni coordinator for her university, where she organizes reunions and maintains connections among graduates. Her work is driven by a passion for building community and supporting educational initiatives. Her home on Pine St serves as a venue for alumni gatherings, where stories are shared and new projects are born.
    """,
    """
    Steve Adams is a personal trainer who owns a popular gym in Muscleville. He is committed to helping his clients achieve their fitness goals through tailored workout plans and motivational coaching. His home on Fir Ave is equipped with a small gym where he develops new fitness programs and trains for marathons.
    """,
    """
    Tina Nguyen is a marketing executive who specializes in customer loyalty programs. She is known for her creative campaigns that boost brand loyalty and customer engagement. Her home in Marketcity is stylishly decorated, showcasing her keen eye for design. Tina also volunteers for local business workshops, sharing her expertise with small business owners.
    """
]

df = spark.createDataFrame(
    pd.DataFrame(
        {
            "unmasked_text": sentences
        }
    )
)
display(df)

# COMMAND ----------

_ = (
  df.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.backstories")
)
