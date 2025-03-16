module WebAppSample1.Dto

open System
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Mvc

type UserDto = { Id: int; Name: string }

[<CLIMutable>]
type CustomerRequestModel = { Id: int }

[<CLIMutable>]
type SearchCustomerRequestModel =
    { NamePattern: string
      
      [<FromServices>] HttpContext: HttpContext }

type Customer = { Id: int; Name: string; Birthday: DateOnly }

[<CLIMutable>]
type UserRequestModel = { Id: int }
