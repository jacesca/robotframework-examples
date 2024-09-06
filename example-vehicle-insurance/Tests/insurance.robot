*** Settings ***
Library    Browser
Library    DateTime
Library    OperatingSystem


*** Variables ***
${BROWSER}    chromium
${HEADLESS}   false


*** Test Cases ***
Create Quote for Car
    Open Insurance Application
    Enter Vehicle Data for Automobile
    Enter Insurance Data
    Enter Product Data
    Enter Price Option
    Send Quote
    End Test


*** Keywords ***
Open Insurance Application
   New Browser    browser=${BROWSER}    headless=${HEADLESS}
   New Context    locale=en-GB 
   # When you set the locale to en-GB, the new browser context will behave as 
   # if it's operating in an environment configured for British English. 
   # This can affect how dates, numbers, currency, and text are displayed or 
   # interpreted within the web application being tested.
   New Page    http://sampleapp.tricentis.com/
   Take Screenshot    home-site

End Test
    Close Context
    Close Browser
    

Enter Vehicle Data for Automobile
    Click    div.main-navigation >> "Automobile"
    Log    Get Url
    Wait For Load State
    Select Options By    id=make    text    Audi
    Fill Text    id=engineperformance    110
    Fill Text    id=dateofmanufacture    06/12/1980
    Select Options By    id=numberofseats    text    5
    Select Options By    id=fuel    text    Petrol
    Fill Text    id=listprice    30000
    Fill Text    id=licenseplatenumber    DMK1234
    Fill Text    id=annualmileage    10000
    Take Screenshot    vehicle-data
    # Click    id=nextenterinsurantdata    left  # Another way to do the next line
    Click    section[style="display: block;"] >> text=Next »
    Press Keys    id=nextenterinsurantdata    Enter
    # There are multiple buttons with `text=Next »` label but only one visible
    # at a time. The `style="display: block;"` will find the one that is 
    # visible.

Enter Insurance Data
    [Arguments]    ${firstname}=Max    ${Lastname}=Mustermann
    Wait For Load State
    Fill Text    id=firstname    ${firstname}
    Fill Text    id=lastname    ${Lastname}
    Fill Text    id=birthdate    01/31/1980
    Check Checkbox    *css=label >> id=gendermale
    Fill Text    id=streetaddress    Test Street
    Select Options By    id=country    text    Germany
    Fill Text    id=zipcode    40123
    Fill Text    id=city    Essen
    Select Options By    id=occupation    text    Employee
    Click    text=Cliff Diving
    Take Screenshot    insurance-vehicle
    Click    section[style="display: block;"] >> text=Next »
    
Enter Product Data
    ${OneMonthAhead} =    Get Current Date    result_format=%m/%d/%Y    increment=31 days  # Valid formats: https://robotframework.org/robotframework/latest/libraries/DateTime.html#Time%20formats
    Log    ${OneMonthAhead}
    # ${None}: Passing ${None} as the first argument tells Subtract Time 
    # to use the current date and time.
    Wait For Load State
    Fill Text    id=startdate    ${OneMonthAhead}  # Month/Day/Year
    Select Options By    id=insurancesum    text    7.000.000,00
    Select Options By    id=meritrating    text    Bonus 1
    Select Options By    id=damageinsurance    text    No Coverage
    Check Checkbox    *css=label >> id=EuroProtection
    Select Options By    id=courtesycar    text    Yes
    Take Screenshot    product-data
    Click    section[style="display: block;"] >> text=Next »

Enter Price Option
    [Arguments]    ${price_option}=Silver
    Wait For Load State
    Click    *css=label >> css=[value=${price_option}]
    Take Screenshot    price-data
    Click    section[style="display: block;"] >> text=Next »

Send Quote
    [Arguments]    ${file_name}=Quote-file.pdf
    Wait For Load State
    Fill Text   "E-Mail" >> .. >> input    max.mustermann@example.com
    Fill Text   "Phone" >> .. >> input    0049201123456
    Fill Text   "Username" >> .. >> input    max.mustermann
    Fill Text   "Password" >> .. >> input    SecretPassword123!
    Fill Text   "Confirm Password" >> .. >> input    SecretPassword123!
    Fill Text   "Comments" >> .. >> textarea    Some comments
    Click    "« Send »"
    Take Screenshot    send-quote
    Wait For Elements State    "Sending e-mail success!"    timeout=50s
    Click    "OK"
