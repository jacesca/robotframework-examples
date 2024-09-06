*** Settings ***
Library    OperatingSystem


*** Variables ***
&{MY-NEW-DICT}                username=demo    password=mode
&{MY-NEW-DICT2}                username=demo2    password=mode2


*** Keywords ***
Log My Username
    [Arguments]    ${USERNAME}
    Log            ${USERNAME}

Log My Password
    [Arguments]    ${PASSWORD}
    Log            ${PASSWORD}
    
Log My Username and Password
    Log My Username    ${MY-NEW-DICT}[username]
    Log My Password    ${MY-NEW-DICT}[password]

Log Specific Username and Password
    [Arguments]        ${USERNAME}    ${PASSWORD}
    Log My Username    ${USERNAME}
    Log My Password    ${PASSWORD}


*** Test Cases ***
TEST7
    Log My Username    ${MY-NEW-DICT}[username]
    Log My Password    ${MY-NEW-DICT}[password]

TEST8
    Log My Username and Password

TEST9
    Log Specific Username and Password    ${MY-NEW-DICT}[username]    ${MY-NEW-DICT}[password]
