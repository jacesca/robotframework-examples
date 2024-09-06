*** Settings ***
Documentation     Simple example using SeleniumLibrary. 
Library           SeleniumLibrary


*** Variables ***
${Site-Url}    http://localhost:8040/WebDemo/html/
${Browser}     chrome


*** Test Cases ***
Valid Login
    Open Browser To Login Page
    Type In Username    demo
    Type In Password    mode
    Sleep    3s    # Not recommendable to use, only when debugging
    Submit Credentials
    Welcome Page should be Open
    Sleep    3s    # Not recommendable to use, only when debugging
    # Teardown is executed no matter if success or fail
    [Teardown]    Close Browser     # SeleniumLibrary command


*** Keywords *** 
Open Browser To Login Page
    # TODO: implement keyword "Open Browser To Login Page".
    Open Browser    ${Site-Url}    ${Browser}
    Title Should Be    Login Page     # SeleniumLibrary command
    

Type In Username
    [Arguments]    ${username}
    # TODO: implement keyword "Type In Username".
    Input Text    id=username_field    ${username}


Type In Password
    [Arguments]    ${password}
    # TODO: implement keyword "Type In Password".
    Input Text    id=password_field    ${password}


Submit Credentials
    # TODO: implement keyword "Submit Credentials".
    Click Button    id=login_button


Welcome Page should be Open
    # TODO: implement keyword "Welcome Page should be Open".
    Title Should Be    Welcome Page
