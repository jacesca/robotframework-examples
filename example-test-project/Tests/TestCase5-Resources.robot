*** Settings ***
Documentation       How to use resources example
Resource            ../Resources/resources.robot


*** Test Cases ***
TEST10
    Log Specific Username and Password    ${MY-NEW-DICT}[username]    ${MY-NEW-DICT}[password]