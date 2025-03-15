module WebAppSample1.Dto

open System

type UserDto = { Id: int; Name: string }

[<CLIMutable>]
type CustomerRequestModel = { Id: int }

[<CLIMutable>]
type SearchCustomerRequestModel = { NamePattern: string }

type Customer = { Id: int; Name: string; Birthday: DateOnly }

[<CLIMutable>]
type UserRequestModel = { Id: int }
