*** Settings ***
Library    OperatingSystem

*** Variables ***
${MY-VAR-1}               My test variable 1
${MY-VAR-2}               My test variable 2
${GOOGLE-SEARCH-FIELD}    //input[@title="Search"]

@{MY-LIST}                test1    test2    test3    test4    test5    test6

&{MY-DICT}                firstname=demo    lastname=mode

*** Test Cases ***
TEST6
    [Tags]   Demo
    Log    ${MY-VAR-1}
    Log    ${MY-VAR-2}
    Log    ${GOOGLE-SEARCH-FIELD}
    Log    ${MY-LIST}[2]
    Log    ${MY-DICT}[firstname]
